import cv2
import os
import argparse
from bike_detector import BikeDetector
from trick_classifier import TrickClassifier


class BikeManeuverDetector:
    def __init__(self, yolo_model='yolov8m.pt', classifier_model='trick_classifier_model.h5'):
        print("Inicializando detector de bicicletas...")
        self.bike_detector = BikeDetector(model_path=yolo_model, confidence_threshold=0.5)
        
        print("Carregando classificador de manobras...")    
        self.trick_classifier = TrickClassifier()
        
        if os.path.exists(classifier_model):
            self.trick_classifier.load_model(classifier_model)
            self.classifier_loaded = True
        else:
            print(f"AVISO: Modelo de classificação não encontrado em {classifier_model}")
            print("Execute train_model.py primeiro para treinar o classificador")
            self.classifier_loaded = False
    
    def process_video_realtime(self, video_path, output_path='output_with_tricks.mp4', 
                               window_size=30, stride=10):
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_buffer = {}
        frame_count = 0
        trick_detected_frames = set()
        
        print(f"\nProcessando vídeo: {video_path}")
        print(f"FPS: {fps}, Resolução: {width}x{height}, Total de frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bike_detections = self.bike_detector.detect_bikes(frame)
            
            current_trick = None
            current_confidence = 0.0
            
            if bike_detections and self.classifier_loaded:
                for detection in bike_detections:
                    cropped_bike = self.bike_detector.crop_bike(frame, detection['bbox'])
                    
                    if cropped_bike.size > 0:
                        track_id = detection.get('id', 0)
                        frame_buffer.setdefault(track_id, []).append(cropped_bike)
                        
                        if len(frame_buffer[track_id]) >= window_size:
                            if len(frame_buffer[track_id]) > window_size:
                                frame_buffer[track_id] = frame_buffer[track_id][-window_size:]
                            
                            if frame_count % stride == 0:
                                prediction = self.trick_classifier.predict_sequence(
                                    frame_buffer[track_id], 
                                    aggregate='average'
                                )
                                
                                if prediction['class'] == '360' and prediction['confidence'] > 0.7:
                                    current_trick = '360'
                                    self.last_confidence = prediction['confidence']
                                    
                                    for i in range(max(0, frame_count - window_size), frame_count):
                                        trick_detected_frames.add(i)
            
            annotated_frame = frame.copy()
            
            for detection in bike_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                color = (0, 255, 0)
                thickness = 2
                
                if frame_count in trick_detected_frames:
                    color = (0, 0, 255)
                    thickness = 3
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"Bike: {confidence:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            if frame_count in trick_detected_frames:
                cv2.putText(annotated_frame, "MANOBRA 360 DETECTADA!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                self.last_confidence = 0.0
                self.last_confidence = prediction['confidence']

                if self.last_confidence > 0:
                    cv2.putText(annotated_frame, f"Confianca: {self.last_confidence:.2%}", 
                               (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            out.write(annotated_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processados {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        print(f"\nProcessamento concluído!")
        print(f"Vídeo salvo em: {output_path}")
        print(f"Total de frames com manobra 360 detectada: {len(trick_detected_frames)}")
        
        return len(trick_detected_frames)
    
    def analyze_video_summary(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        frame_count = 0
        bike_detected_count = 0
        trick_360_count = 0
        
        print(f"\nAnalisando vídeo: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bike_detections = self.bike_detector.detect_bikes(frame)
            
            if bike_detections:
                bike_detected_count += 1
                
                if self.classifier_loaded:
                    for detection in bike_detections:
                        cropped_bike = self.bike_detector.crop_bike(frame, detection['bbox'])
                        
                        if cropped_bike.size > 0:
                            prediction = self.trick_classifier.predict(cropped_bike)
                            
                            if prediction['class'] == '360' and prediction['confidence'] > 0.7:
                                trick_360_count += 1
                                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Analisados {frame_count}/{total_frames} frames")
        
        cap.release()
        
        print("\n" + "="*50)
        print("RESUMO DA ANÁLISE")
        print("="*50)
        print(f"Total de frames: {total_frames}")
        print(f"Duração: {total_frames/fps:.2f} segundos")
        print(f"Frames com bicicleta detectada: {bike_detected_count}")
        print(f"Frames com manobra 360 detectada: {trick_360_count}")
        print(f"Porcentagem de frames com bicicleta: {bike_detected_count/total_frames*100:.2f}%")
        
        if bike_detected_count > 0:
            print(f"Porcentagem de manobra 360: {trick_360_count/bike_detected_count*100:.2f}%")
        
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Detector de manobras de bicicleta')
    parser.add_argument('--video', type=str, default='videos/crianca_bicicleta.mp4',
                       help='Caminho para o vídeo de entrada')
    parser.add_argument('--output', type=str, default='output_with_tricks.mp4',
                       help='Caminho para o vídeo de saída')
    parser.add_argument('--yolo_model', type=str, default='yolov8m.pt',
                       help='Modelo YOLO a ser usado')
    parser.add_argument('--classifier_model', type=str, default='trick_classifier_model.h5',
                       help='Modelo de classificação de manobras')
    parser.add_argument('--mode', type=str, choices=['detect', 'analyze', 'extract'], 
                       default='detect',
                       help='Modo de operação: detect (processar vídeo), analyze (análise resumida), extract (extrair frames)')
    parser.add_argument('--window_size', type=int, default=30,
                       help='Tamanho da janela para análise temporal')
    parser.add_argument('--stride', type=int, default=10,
                       help='Stride para análise temporal')
    
    args = parser.parse_args()
    
    if args.mode == 'extract':
        detector = BikeDetector(model_path=args.yolo_model, confidence_threshold=0.5)
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
                stride=args.stride
            )
        
        elif args.mode == 'analyze':
            detector.analyze_video_summary(args.video)


if __name__ == "__main__":
    main()
