import cv2
import numpy as np
from ultralytics import YOLO
import os


class BikeDetector:
    def __init__(self, model_path='yolov8m.pt', confidence_threshold=0.4):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.bike_class_id = 1
        self.track_history = {}      # track_id -> lista de detecções REAIS {'frame','bbox','confidence'}
        self.last_seen = {}          # track_id -> último frame com detecção REAL
        self.max_frames_missing = 5
        self.max_history = 30        # limite de detecções guardadas por track

    def detect_bikes(self, frame, frame_number=0):
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            imgsz=1280,
            conf=self.confidence_threshold,
            iou=0.5,
            tracker='bytetrack.yaml'
        )

        height, width = frame.shape[:2]
        bike_detections = []
        current_frame_ids = set()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if class_id == self.bike_class_id and confidence >= self.confidence_threshold:
                    # Sem track id confirmado ainda: ignorar para não fundir bikes
                    # distintas no id 0 (corromperia histórico, interpolação e buffer do LSTM)
                    if box.id is None:
                        continue

                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    track_id = int(box.id[0])

                    bbox = [int(x1), int(y1), int(x2), int(y2)]

                    bike_detections.append({
                        'bbox': bbox,
                        'confidence': confidence,
                        'id': track_id,
                        'interpolated': False
                    })

                    current_frame_ids.add(track_id)

                    self.track_history.setdefault(track_id, []).append({
                        'frame': frame_number,
                        'bbox': bbox,
                        'confidence': confidence
                    })
                    if len(self.track_history[track_id]) > self.max_history:
                        self.track_history[track_id] = self.track_history[track_id][-self.max_history:]

                    self.last_seen[track_id] = frame_number

        interpolated_detections = self._interpolate_missing_tracks(
            frame_number, current_frame_ids, (height, width)
        )
        bike_detections.extend(interpolated_detections)

        self._prune_dead_tracks(frame_number)

        return bike_detections

    def _interpolate_missing_tracks(self, current_frame, detected_ids, frame_shape):
        interpolated = []
        height, width = frame_shape

        for track_id, history in self.track_history.items():
            if track_id in detected_ids:
                continue

            if len(history) < 2:
                continue

            # frames_missing é medido a partir da última detecção REAL.
            # Como detecções interpoladas NÃO entram no histórico, esse contador
            # cresce a cada frame ausente e o teto max_frames_missing realmente dispara.
            last_real = history[-1]
            frames_missing = current_frame - self.last_seen.get(track_id, last_real['frame'])

            if not (0 < frames_missing <= self.max_frames_missing):
                continue

            prev_detection = history[-2]
            # Guarda contra dt~0 (duas detecções no mesmo frame): velocidade explodiria
            if last_real['frame'] == prev_detection['frame']:
                continue
            dt = float(last_real['frame'] - prev_detection['frame'])
            dx = (last_real['bbox'][0] - prev_detection['bbox'][0]) / dt
            dy = (last_real['bbox'][1] - prev_detection['bbox'][1]) / dt
            dw = (last_real['bbox'][2] - prev_detection['bbox'][2]) / dt
            dh = (last_real['bbox'][3] - prev_detection['bbox'][3]) / dt

            new_x1 = int(last_real['bbox'][0] + dx * frames_missing)
            new_y1 = int(last_real['bbox'][1] + dy * frames_missing)
            new_x2 = int(last_real['bbox'][2] + dw * frames_missing)
            new_y2 = int(last_real['bbox'][3] + dh * frames_missing)

            # Manter a box dentro do frame
            new_x1 = max(0, min(new_x1, width - 1))
            new_x2 = max(0, min(new_x2, width - 1))
            new_y1 = max(0, min(new_y1, height - 1))
            new_y2 = max(0, min(new_y2, height - 1))

            # Descartar boxes degeneradas (a bike já saiu do frame)
            if new_x2 - new_x1 < 5 or new_y2 - new_y1 < 5:
                continue

            interpolated.append({
                'bbox': [new_x1, new_y1, new_x2, new_y2],
                'confidence': max(0.3, last_real['confidence'] - 0.1 * frames_missing),
                'id': track_id,
                'interpolated': True
            })

        return interpolated

    def _prune_dead_tracks(self, current_frame):
        """Remove tracks que sumiram há mais que max_frames_missing frames.

        Sem isso, históricos antigos se acumulam e o tracker continua tentando
        interpolar boxes-fantasma que já saíram de cena.
        """
        dead_ids = [
            track_id for track_id, last_frame in self.last_seen.items()
            if current_frame - last_frame > self.max_frames_missing
        ]
        for track_id in dead_ids:
            self.track_history.pop(track_id, None)
            self.last_seen.pop(track_id, None)
    
    def crop_bike(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        if x2 - x1 < 64 or y2 - y1 < 64:
            return np.array([])
        cropped = frame[y1:y2, x1:x2]
        return cropped
    
    def process_video(self, video_path, output_dir='output_frames'):
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processando vídeo: {video_path}")
        print(f"FPS: {fps}, Total de frames: {total_frames}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bike_detections = self.detect_bikes(frame, frame_number=frame_count)
            
            if bike_detections:
                for idx, detection in enumerate(bike_detections):
                    cropped_bike = self.crop_bike(frame, detection['bbox'])
                    
                    if cropped_bike.size > 0:
                        is_interpolated = detection.get('interpolated', False)
                        suffix = '_interp' if is_interpolated else ''
                        output_path = os.path.join(
                            output_dir, 
                            f"bike_frame_{frame_count:06d}_det_{idx}{suffix}.jpg"
                        )
                        cv2.imwrite(output_path, cropped_bike)
                        saved_count += 1
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processados {frame_count}/{total_frames} frames, {saved_count} bicicletas detectadas")
        
        cap.release()
        print(f"\nProcessamento concluído!")
        print(f"Total de frames processados: {frame_count}")
        print(f"Total de bicicletas detectadas e salvas: {saved_count}")
        
        return saved_count
    
    def process_video_with_visualization(self, video_path, output_video_path='output_video.mp4'):
        cap = cv2.VideoCapture(video_path)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            bike_detections = self.detect_bikes(frame, frame_number=frame_count)
            
            for detection in bike_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                is_interpolated = detection.get('interpolated', False)
                
                color = (0, 165, 255) if is_interpolated else (0, 255, 0)
                thickness = 2 if is_interpolated else 2
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                label = f"Bike: {confidence:.2f}"
                if is_interpolated:
                    label += " (interp)"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processados {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Vídeo salvo em: {output_video_path}")


if __name__ == "__main__":
    detector = BikeDetector(confidence_threshold=0.6)
    
    video_path = "videos/crianca_bicicleta.mp4"
    
    if os.path.exists(video_path):
        print("Extraindo frames das bicicletas detectadas...")
        detector.process_video(video_path, output_dir='bike_frames')
        
        print("\nCriando vídeo com detecções visualizadas...")
        detector.process_video_with_visualization(video_path, output_video_path='output_detected.mp4')
    else:
        print(f"Vídeo não encontrado: {video_path}")
