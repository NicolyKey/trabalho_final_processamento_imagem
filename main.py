import cv2
import os
import argparse
from collections import deque
from bike_detector import BikeDetector
from sequence_trick_classifier import SequenceTrickClassifier


class BikeManeuverDetector:
    def __init__(self, yolo_model='yolov8m.pt',
                 classifier_model='sequence_trick_classifier_cnn_lstm.h5'):
        print("Inicializando detector de bicicletas...")
        self.bike_detector = BikeDetector(model_path=yolo_model, confidence_threshold=0.4)

        print("Carregando classificador de sequência (CNN+LSTM)...")
        self.classifier = SequenceTrickClassifier()

        if os.path.exists(classifier_model):
            self.classifier.load_model(classifier_model)
            self.classifier_loaded = True
        else:
            print(f"AVISO: Modelo de sequência não encontrado em {classifier_model}")
            print("Execute train_sequence_model.py primeiro para treinar o classificador")
            self.classifier_loaded = False

    def _evaluate_track(self, state, buffer):
        """Roda o LSTM na janela de frames de um track e atualiza seu estado com histerese.

        Histerese: precisa de confiança alta (start_threshold) para LIGAR o 360 e
        ela precisa cair abaixo de end_threshold para DESLIGAR. Isso evita flicker e,
        principalmente, permite detectar o FIM da manobra quando o modelo volta a
        classificar a sequência como 'normal'.
        """
        prediction = self.classifier.predict_sequence(list(buffer))
        conf_360 = prediction['all_predictions'].get('360', 0.0)

        # Debounce: conta janelas consecutivas. Exige rotação sustentada
        # (debounce_on janelas) para LIGAR — mata o glimpse de 1 janela — e
        # queda sustentada (debounce_off) para DESLIGAR.
        if conf_360 >= state['start_threshold']:
            state['consec_on'] += 1
            state['consec_off'] = 0
        elif conf_360 < state['end_threshold']:
            state['consec_off'] += 1
            state['consec_on'] = 0
        else:
            state['consec_on'] = 0

        if not state['is_360'] and state['consec_on'] >= state['debounce_on']:
            state['is_360'] = True
        elif state['is_360'] and state['consec_off'] >= state['debounce_off']:
            state['is_360'] = False

        state['confidence'] = conf_360
        return conf_360

    def process_video_realtime(self, video_path, output_path='output_with_tricks.mp4',
                               window_size=30, stride=5,
                               start_threshold=0.6, end_threshold=0.45,
                               debounce_on=2, debounce_off=1):
        cap = cv2.VideoCapture(video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Garante que a pasta de saída exista (ex.: resultados/)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        seq_len = self.classifier.sequence_length if self.classifier_loaded else 15
        # A janela precisa ter pelo menos seq_len frames para uma predição significativa
        window_size = max(window_size, seq_len)

        track_buffers = {}   # track_id -> deque de crops RGB (janela deslizante)
        track_states = {}    # track_id -> estado da manobra (histerese)
        frame_count = 0
        frames_com_360 = 0

        print(f"\nProcessando vídeo: {video_path}")
        print(f"FPS: {fps}, Resolução: {width}x{height}, Total de frames: {total_frames}")
        if not self.classifier_loaded:
            print("AVISO: rodando sem classificador — apenas detecção/tracking de bikes.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.bike_detector.detect_bikes(frame, frame_number=frame_count)

            # --- Atualizar buffers e estados por track ---
            if self.classifier_loaded:
                for detection in detections:
                    # Não alimentar o classificador com boxes interpoladas (crop drifta / vazio)
                    if detection.get('interpolated', False):
                        continue

                    track_id = detection.get('id', 0)
                    cropped = self.bike_detector.crop_bike(frame, detection['bbox'])
                    if cropped.size == 0:
                        continue

                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    track_buffers.setdefault(track_id, deque(maxlen=window_size)).append(cropped_rgb)
                    state = track_states.setdefault(track_id, {
                        'is_360': False,
                        'confidence': 0.0,
                        'start_threshold': start_threshold,
                        'end_threshold': end_threshold,
                        'debounce_on': debounce_on,
                        'debounce_off': debounce_off,
                        'consec_on': 0,
                        'consec_off': 0,
                    })

                    if len(track_buffers[track_id]) >= seq_len and frame_count % stride == 0:
                        conf_360 = self._evaluate_track(state, track_buffers[track_id])
                        if os.environ.get('VERBOSE_CONF'):
                            print(f"  [CONF] frame={frame_count} id={track_id} "
                                  f"conf_360={conf_360:.3f} is_360={state['is_360']}")
                        elif frame_count % 30 == 0:
                            print(f"  [DEBUG] frame={frame_count} id={track_id} "
                                  f"conf_360={conf_360:.3f} is_360={state['is_360']}")

                # Limpar buffers/estados de tracks que o detector já descartou
                alive_ids = set(self.bike_detector.last_seen.keys())
                for tid in list(track_buffers.keys()):
                    if tid not in alive_ids:
                        track_buffers.pop(tid, None)
                        track_states.pop(tid, None)

            # --- Desenhar ---
            annotated_frame = frame.copy()
            frame_has_360 = False

            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                track_id = detection.get('id', 0)
                state = track_states.get(track_id)
                is_360 = bool(state and state['is_360'])

                if is_360:
                    frame_has_360 = True
                    color = (0, 0, 255)
                    thickness = 3
                    label = f"360! ({state['confidence']:.0%})"
                else:
                    color = (0, 255, 0)
                    thickness = 2
                    label = f"Bike: {detection['confidence']:.2f}"
                    if detection.get('interpolated', False):
                        label += " (interp)"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if frame_has_360:
                frames_com_360 += 1
                cv2.putText(annotated_frame, "MANOBRA 360 DETECTADA", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            out.write(annotated_frame)
            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Processados {frame_count}/{total_frames} frames")

        cap.release()
        out.release()

        print(f"\nProcessamento concluído!")
        print(f"Vídeo salvo em: {output_path}")
        print(f"Total de frames com manobra 360 detectada: {frames_com_360}")

        return frames_com_360

    def analyze_video_summary(self, video_path, window_size=30, stride=5,
                              start_threshold=0.6, end_threshold=0.45,
                              debounce_on=2, debounce_off=1):
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        seq_len = self.classifier.sequence_length if self.classifier_loaded else 15
        window_size = max(window_size, seq_len)

        track_buffers = {}
        track_states = {}
        frame_count = 0
        bike_detected_count = 0
        frames_com_360 = 0

        print(f"\nAnalisando vídeo: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = self.bike_detector.detect_bikes(frame, frame_number=frame_count)

            if detections:
                bike_detected_count += 1

            if self.classifier_loaded:
                for detection in detections:
                    if detection.get('interpolated', False):
                        continue
                    track_id = detection.get('id', 0)
                    cropped = self.bike_detector.crop_bike(frame, detection['bbox'])
                    if cropped.size == 0:
                        continue

                    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                    track_buffers.setdefault(track_id, deque(maxlen=window_size)).append(cropped_rgb)
                    state = track_states.setdefault(track_id, {
                        'is_360': False,
                        'confidence': 0.0,
                        'start_threshold': start_threshold,
                        'end_threshold': end_threshold,
                        'debounce_on': debounce_on,
                        'debounce_off': debounce_off,
                        'consec_on': 0,
                        'consec_off': 0,
                    })

                    if len(track_buffers[track_id]) >= seq_len and frame_count % stride == 0:
                        self._evaluate_track(state, track_buffers[track_id])

                alive_ids = set(self.bike_detector.last_seen.keys())
                for tid in list(track_buffers.keys()):
                    if tid not in alive_ids:
                        track_buffers.pop(tid, None)
                        track_states.pop(tid, None)

            # Contar pelo estado dos tracks (cobre gaps de interpolação, igual ao realtime)
            frame_has_360 = any(s['is_360'] for s in track_states.values())
            if frame_has_360:
                frames_com_360 += 1

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Analisados {frame_count}/{total_frames} frames")

        cap.release()

        print("=" * 50)
        print(f"Frames com bike detectada: {bike_detected_count}/{frame_count}")
        if frame_count > 0:
            print(f"Frames com manobra 360: {frames_com_360} ({frames_com_360 / frame_count * 100:.2f}%)")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Detector de manobras de bicicleta')
    parser.add_argument('--video', type=str, default='videos/crianca_bicicleta.mp4',
                       help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', type=str, default='resultados/resultado.mp4',
                       help='Caminho para o vídeo de saída (pasta criada automaticamente)')
    parser.add_argument('--yolo_model', type=str, default='yolov8m.pt',
                       help='Modelo YOLO a ser usado')
    parser.add_argument('--classifier_model', type=str,
                       default='sequence_trick_classifier_cnn_lstm.h5',
                       help='Modelo de classificação de manobras (sequência CNN+LSTM)')
    parser.add_argument('--mode', type=str, choices=['detect', 'analyze', 'extract'],
                       default='detect',
                       help='Modo: detect (processar vídeo), analyze (resumo), extract (extrair frames)')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Tamanho da janela deslizante de frames por bike')
    parser.add_argument('--stride', type=int, default=5,
                       help='De quantos em quantos frames rodar o classificador')
    parser.add_argument('--start_threshold', type=float, default=0.6,
                       help='Confiança mínima de 360 para INICIAR a marcação da manobra')
    parser.add_argument('--end_threshold', type=float, default=0.45,
                       help='Confiança de 360 abaixo da qual a manobra é considerada ENCERRADA')
    parser.add_argument('--debounce_on', type=int, default=2,
                       help='Janelas consecutivas de rotação necessárias para INICIAR a marcação')
    parser.add_argument('--debounce_off', type=int, default=1,
                       help='Janelas consecutivas abaixo do limiar para ENCERRAR a marcação')

    args = parser.parse_args()

    if args.mode == 'extract':
        detector = BikeDetector(model_path=args.yolo_model, confidence_threshold=0.4)
        detector.process_video(args.video, output_dir=f'bike_frames/{args.video}')
        detector.process_video_with_visualization(args.video, output_video_path=f'output/{args.video}')

    else:
        detector = BikeManeuverDetector(
            yolo_model=args.yolo_model,
            classifier_model=args.classifier_model
        )

        if args.mode == 'detect':
            detector.process_video_realtime(
                args.video,
                args.output,
                window_size=args.window_size,
                stride=args.stride,
                start_threshold=args.start_threshold,
                end_threshold=args.end_threshold,
                debounce_on=args.debounce_on,
                debounce_off=args.debounce_off
            )

        elif args.mode == 'analyze':
            detector.analyze_video_summary(
                args.video,
                window_size=args.window_size,
                stride=args.stride,
                start_threshold=args.start_threshold,
                end_threshold=args.end_threshold,
                debounce_on=args.debounce_on,
                debounce_off=args.debounce_off
            )


if __name__ == "__main__":
    main()
