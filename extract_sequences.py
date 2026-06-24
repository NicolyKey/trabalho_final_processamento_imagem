"""Extrai sequências de crops de bicicleta a partir de vídeos brutos.

Para a classe 360, faz AUTO-RECORTE: dentro do track, localiza o trecho de
maior mudança visual (a rotação) e salva SÓ esse núcleo como sequência de 360.
Os frames de aproximação/saída (a bike só pedalando) viram NEGATIVOS (classe
normal) — hard negatives no mesmo contexto, que ensinam o modelo a não marcar
pedalada como 360.

Para a classe normal, fatia o track inteiro em janelas.

Uso:
  python extract_sequences.py --label 360 --videos videos/360_de_frente.mp4 ...
  python extract_sequences.py --label normal --videos videos/andando_normal_de_lado.mp4 ...
"""
import cv2
import os
import argparse
import numpy as np
from bike_detector import BikeDetector


def _collect_track_crops(video_path, yolo_model, max_frames):
    detector = BikeDetector(model_path=yolo_model, confidence_threshold=0.4)
    cap = cv2.VideoCapture(video_path)
    track_crops = {}
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break
        for d in detector.detect_bikes(frame, frame_number=frame_count):
            if d.get('interpolated', False):
                continue
            crop = detector.crop_bike(frame, d['bbox'])
            if crop.size == 0:
                continue
            track_crops.setdefault(d['id'], []).append(crop)
        frame_count += 1
    cap.release()
    return track_crops, frame_count


def _motion_signal(crops):
    """Mudança visual frame-a-frame do conteúdo do crop (rotação => alta)."""
    small = []
    for c in crops:
        g = cv2.cvtColor(cv2.resize(c, (64, 64)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        small.append(g)
    motion = [0.0]
    for i in range(1, len(small)):
        motion.append(float(np.mean(np.abs(small[i] - small[i - 1]))) / 255.0)
    if len(motion) > 1:
        motion[0] = motion[1]
    return np.array(motion)


def _find_core(crops, core_len):
    """Janela contígua de maior movimento acumulado (o giro)."""
    motion = _motion_signal(crops)
    n = len(motion)
    L = min(core_len, n)
    if L >= n:
        return 0, n
    sums = np.convolve(motion, np.ones(L), mode='valid')
    start = int(np.argmax(sums))
    return start, start + L


def _slice_windows(crops, window, stride, min_frames):
    if len(crops) < min_frames:
        return []
    if len(crops) <= window:
        return [crops]
    windows = []
    i = 0
    while i + min_frames <= len(crops):
        windows.append(crops[i:i + window])
        i += stride
    return windows


def _save_windows(windows, out_dir, prefix, start_idx, min_frames):
    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for w in windows:
        if len(w) < min_frames:
            continue
        count += 1
        seq_dir = os.path.join(out_dir, f"{prefix}_seq{start_idx + count:03d}")
        os.makedirs(seq_dir, exist_ok=True)
        for j, c in enumerate(w):
            cv2.imwrite(os.path.join(seq_dir, f"frame_{j:04d}.jpg"), c)
    return count


def extract_from_video(video_path, label, output_root, yolo_model='yolov8m.pt',
                       window=20, stride=5, min_frames=15, max_frames=300,
                       core_len=35):
    if not os.path.exists(video_path):
        print(f"  [SKIP] vídeo não encontrado: {video_path}")
        return 0, 0

    track_crops, frame_count = _collect_track_crops(video_path, yolo_model, max_frames)
    base = os.path.splitext(os.path.basename(video_path))[0]

    dir_360 = os.path.join(output_root, '360')
    dir_normal = os.path.join(output_root, 'normal')

    pos_total = 0
    neg_total = 0

    for tid, crops in sorted(track_crops.items(), key=lambda kv: -len(kv[1])):
        if len(crops) < min_frames:
            continue

        if label == '360':
            # Núcleo de rotação -> positivos; aproximação/saída -> negativos
            s, e = _find_core(crops, core_len)
            core = crops[s:e]
            pre = crops[:s]
            post = crops[e:]

            pos_total += _save_windows(
                _slice_windows(core, window, stride, min_frames),
                dir_360, f"{base}_t{tid}", pos_total, min_frames)

            for run in (pre, post):
                neg_total += _save_windows(
                    _slice_windows(run, window, stride, min_frames),
                    dir_normal, f"{base}_t{tid}_ride", neg_total, min_frames)
        else:
            neg_total += _save_windows(
                _slice_windows(crops, window, stride, min_frames),
                dir_normal, f"{base}_t{tid}", neg_total, min_frames)

    if label == '360':
        print(f"  [360] {base}: {frame_count} frames -> {pos_total} seq 360 (giro) + {neg_total} seq normal (pedalada)")
    else:
        print(f"  [normal] {base}: {frame_count} frames -> {neg_total} sequências")
    return pos_total, neg_total


def main():
    parser = argparse.ArgumentParser(description='Extrair sequências de crops de bicicleta de vídeos')
    parser.add_argument('--label', type=str, required=True, help='Classe (360 ou normal)')
    parser.add_argument('--videos', type=str, nargs='+', required=True, help='Lista de vídeos')
    parser.add_argument('--output_root', type=str, default='sequences_dataset')
    parser.add_argument('--yolo_model', type=str, default='yolov8m.pt')
    parser.add_argument('--window', type=int, default=20, help='Frames por sequência')
    parser.add_argument('--stride', type=int, default=5, help='Passo entre janelas')
    parser.add_argument('--min_frames', type=int, default=15, help='Mínimo de frames por sequência')
    parser.add_argument('--max_frames', type=int, default=300, help='Máx. frames lidos por vídeo')
    parser.add_argument('--core_len', type=int, default=35,
                        help='Tamanho do núcleo de rotação extraído como 360 (frames)')

    args = parser.parse_args()

    print(f"Extraindo sequências da classe '{args.label}' de {len(args.videos)} vídeo(s)...")
    pos, neg = 0, 0
    for video in args.videos:
        p, n = extract_from_video(
            video, args.label, args.output_root,
            yolo_model=args.yolo_model, window=args.window, stride=args.stride,
            min_frames=args.min_frames, max_frames=args.max_frames, core_len=args.core_len)
        pos += p
        neg += n

    if args.label == '360':
        print(f"\nTotal: {pos} sequências 360 (giro) + {neg} sequências normal (pedalada extraídas dos clipes de 360)")
    else:
        print(f"\nTotal de sequências '{args.label}': {neg}")


if __name__ == "__main__":
    main()
