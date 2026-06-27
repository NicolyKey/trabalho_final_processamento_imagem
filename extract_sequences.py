"""
Uso:
    python extract_sequences.py                 # processa todos os videos de videos/
    python extract_sequences.py 360_estavel.mp4 # processa apenas um video
"""
import sys
import cv2

from config import (
    VIDEOS_DIR, SEQUENCES_DIR, IMG_SIZE, YOLO_WEIGHTS, YOLO_CONF,
    COCO_BICYCLE, COCO_PERSON, CROP_MARGIN,
)


def class_for_video(name: str) -> str:
    return "360" if "360" in name.lower() else "normal"


def best_detection(result):
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_bike = None
    best_bike_conf = -1.0
    best_person = None
    best_person_conf = -1.0

    for i in range(len(boxes)):
        cls = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        if cls == COCO_BICYCLE and conf > best_bike_conf:
            best_bike_conf, best_bike = conf, xyxy
        elif cls == COCO_PERSON and conf > best_person_conf:
            best_person_conf, best_person = conf, xyxy

    return best_bike if best_bike is not None else best_person


def crop_with_margin(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * CROP_MARGIN))
    y1 = max(0, int(y1 - bh * CROP_MARGIN))
    x2 = min(w, int(x2 + bw * CROP_MARGIN))
    y2 = min(h, int(y2 + bh * CROP_MARGIN))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def next_seq_index(class_dir, video_stem) -> int:
    idx = 1
    while (class_dir / f"{video_stem}_seq{idx:03d}").exists():
        idx += 1
    return idx


def extract_video(model, video_path):
    cls = class_for_video(video_path.name)
    class_dir = SEQUENCES_DIR / cls
    class_dir.mkdir(parents=True, exist_ok=True)
    seq_idx = next_seq_index(class_dir, video_path.stem)
    out_dir = class_dir / f"{video_path.stem}_seq{seq_idx:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [erro] nao consegui abrir {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=YOLO_CONF, verbose=False)
        box = best_detection(results[0])
        if box is not None:
            crop = crop_with_margin(frame, box)
            if crop is not None and crop.size > 0:
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                out_path = out_dir / f"bike_frame_{frame_idx:06d}_det_0.jpg"
                cv2.imwrite(str(out_path), crop)
                saved += 1
        frame_idx += 1

    cap.release()
    print(f"  classe={cls:7s} frames_salvos={saved:4d} -> {out_dir.relative_to(SEQUENCES_DIR.parent)}")
    if saved == 0:
        out_dir.rmdir()  # nada detectado, remove pasta vazia
    return saved


def main():
    from ultralytics import YOLO

    print(f"Carregando YOLO ({YOLO_WEIGHTS})...")
    model = YOLO(YOLO_WEIGHTS)

    if len(sys.argv) > 1:
        videos = [VIDEOS_DIR / arg for arg in sys.argv[1:]]
    else:
        videos = sorted(VIDEOS_DIR.glob("*.mp4"))

    print(f"Processando {len(videos)} video(s)...\n")
    total = 0
    for v in videos:
        if not v.exists():
            print(f"  [aviso] video nao encontrado: {v}")
            continue
        print(f"- {v.name}")
        total += extract_video(model, v)
    print(f"\nConcluido. Total de frames recortados: {total}")


if __name__ == "__main__":
    main()
