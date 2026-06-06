import cv2
import numpy as np
from ultralytics import YOLO
import os


class BikeDetector:
    def __init__(self, model_path='yolov8m.pt', confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.bike_class_id = 1
        
    def detect_bikes(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, imgsz=1280)
        
        bike_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                if class_id == self.bike_class_id and confidence >= self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bike_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'id': int(box.id[0]) if box.id is not None else 0
                    })
        
        return bike_detections
    
    def crop_bike(self, frame, bbox):
        x1, y1, x2, y2 = bbox
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
            
            bike_detections = self.detect_bikes(frame)
            
            if bike_detections:
                for idx, detection in enumerate(bike_detections):
                    cropped_bike = self.crop_bike(frame, detection['bbox'])
                    
                    if cropped_bike.size > 0:
                        output_path = os.path.join(
                            output_dir, 
                            f"bike_frame_{frame_count:06d}_det_{idx}.jpg"
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
            
            bike_detections = self.detect_bikes(frame)
            
            for detection in bike_detections:
                x1, y1, x2, y2 = detection['bbox']
                confidence = detection['confidence']
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                label = f"Bike: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            out.write(frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"Processados {frame_count} frames")
        
        cap.release()
        out.release()
        print(f"Vídeo salvo em: {output_video_path}")


if __name__ == "__main__":
    detector = BikeDetector(confidence_threshold=0.5)
    
    video_path = "videos/crianca_bicicleta.mp4"
    
    if os.path.exists(video_path):
        print("Extraindo frames das bicicletas detectadas...")
        detector.process_video(video_path, output_dir='bike_frames')
        
        print("\nCriando vídeo com detecções visualizadas...")
        detector.process_video_with_visualization(video_path, output_video_path='output_detected.mp4')
    else:
        print(f"Vídeo não encontrado: {video_path}")
