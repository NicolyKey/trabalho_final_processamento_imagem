"""
Uso:
    python predict.py videos/360_estavel.mp4
    python predict.py videos/360_estavel.mp4 --out output/resultado.mp4
"""
import argparse
import os
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

from config import (
    MODEL_PATH, CLASSES_PATH, SEQ_LEN, IMG_SIZE, FRAME_STEP, SPAN, OUTPUT_DIR,
    YOLO_WEIGHTS, YOLO_CONF, COCO_BICYCLE, COCO_PERSON, CROP_MARGIN,
)
from extract_sequences import best_detection, crop_with_margin
from motion_features import flow_descriptor, FLOW_SIZE, MOTION_DIM

def load_classes():
    if CLASSES_PATH.exists():
        return CLASSES_PATH.read_text().split()
    from config import CLASSES
    return CLASSES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="caminho do video de entrada")
    ap.add_argument("--out", default=None, help="caminho do video de saida")
    ap.add_argument("--conf", type=float, default=YOLO_CONF, help="confianca YOLO")
    args = ap.parse_args()

    classes = load_classes()
    print(f"Classes: {classes}")

    from ultralytics import YOLO
    print(f"Carregando YOLO ({YOLO_WEIGHTS})...")
    yolo = YOLO(YOLO_WEIGHTS)

    print(f"Carregando modelo CNN-LSTM ({MODEL_PATH})...")
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir o video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = args.out or str(OUTPUT_DIR / f"pred_{os.path.basename(args.video)}")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    window = deque(maxlen=SPAN)      
    grays = deque(maxlen=SPAN)    
    pred_buffer = deque(maxlen=5)
    label, conf = "...", 0.0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = yolo.predict(frame, conf=args.conf, verbose=False)
        box = best_detection(results[0])

        if box is not None:
            crop = crop_with_margin(frame, box)
            if crop is not None and crop.size > 0:
                resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                window.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32))
                grays.append(cv2.resize(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY),
                                        (FLOW_SIZE, FLOW_SIZE))) 

            # Desenha a bbox da bike.
            x1, y1, x2, y2 = map(int, box)
            box_color = (0, 0, 255) if label == "360" else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        if len(window) == SPAN:
            wcrops = list(window)
            wgrays = list(grays)
            clip = np.stack([wcrops[j * FRAME_STEP] for j in range(SEQ_LEN)], axis=0)

            motion = np.zeros((SEQ_LEN, MOTION_DIM), np.float32)
            for j in range(1, SEQ_LEN):
                motion[j] = flow_descriptor(wgrays[(j - 1) * FRAME_STEP],
                                            wgrays[j * FRAME_STEP])
            motion[0] = motion[1]

            probs = model.predict([clip[None, ...], motion[None, ...]], verbose=0)[0]
            pred_buffer.append(probs)
            probs_smooth = np.mean(pred_buffer, axis=0)
            k = int(np.argmax(probs_smooth))
            label, conf = classes[k], float(probs_smooth[k])

        color = (0, 0, 255) if label == "360" else (0, 200, 0)
        text = f"{label} ({conf*100:.0f}%)"
        cv2.rectangle(frame, (0, 0), (360, 40), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}: {text}")

    cap.release()
    writer.release()
    print(f"\nVideo de saida salvo em: {out_path}")


if __name__ == "__main__":
    main()
