"""
Carregamento do dataset de sequencias para o treino.

Estrutura esperada:
    sequences_dataset/
        360/    <seq_dir>/ frame_000.jpg, frame_001.jpg, ...
        normal/ <seq_dir>/ ...

Cada <seq_dir> e uma sequencia ordenada de frames (recortes da bike). Geramos
amostras de tamanho fixo SEQ_LEN a partir de cada sequencia:
  - se a sequencia for mais curta que SEQ_LEN -> reamostragem uniforme (com repeticao);
  - se for >= SEQ_LEN                         -> janelas deslizantes (passo WINDOW_STRIDE).

O split treino/validacao e feito POR SEQUENCIA (nao por janela), de forma
estratificada por classe, para evitar vazamento de frames quase identicos entre
treino e validacao.

A funcao principal `split_sequences()` devolve, para cada split, uma lista de
SequenceSample (frames + janelas de indices + rotulo). O carregamento dos pixels
e a extracao de features ficam a cargo de train.py (para pre-computar features de
cada frame uma unica vez).
"""
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import cv2

from config import (
    SEQUENCES_DIR, CLASSES, SEQ_LEN, IMG_SIZE, FRAME_STEP, SPAN,
    WINDOW_STRIDE, SEQ_CHUNK, VAL_SPLIT, SEED,
)

IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class SequenceSample:
    frames: list          # lista de Paths dos frames, em ordem temporal
    windows: List[list]   # lista de janelas; cada janela e uma lista de indices em `frames`
    label: int            # indice da classe
    group: str = ""       # video de origem (para validacao cruzada agrupada)


def _natural_key(path):
    """Ordena por numero embutido no nome (frame_2 antes de frame_10)."""
    nums = re.findall(r"\d+", path.name)
    return [int(n) for n in nums] if nums else [0]


def _sample_windows(n, seq_len=SEQ_LEN, frame_step=FRAME_STEP,
                    span=SPAN, stride=WINDOW_STRIDE):
    """
    Gera amostras de seq_len indices, cada amostra cobrindo `span` frames de
    origem (1 a cada frame_step). Assim cada amostra abrange a manobra inteira.

    - Se a sequencia for mais curta que `span`: reamostra seq_len frames
      uniformemente por toda a sequencia (captura o que houver da manobra).
    - Caso contrario: desliza uma janela de tamanho `span` (passo `stride`) e,
      dentro dela, pega 1 frame a cada frame_step.
    """
    if n < span:
        idx = np.linspace(0, n - 1, seq_len).round().astype(int)
        return [idx.tolist()]
    windows = []
    start = 0
    while start + span <= n:
        windows.append([start + j * frame_step for j in range(seq_len)])
        start += stride
    last_start = n - span
    last = [last_start + j * frame_step for j in range(seq_len)]
    if not windows or last != windows[-1]:
        windows.append(last)
    return windows


def _chunk(frames, chunk):
    """Divide uma lista longa de frames em pedacos de ate `chunk` frames."""
    if len(frames) <= chunk:
        return [frames]
    n_chunks = round(len(frames) / chunk)
    n_chunks = max(1, n_chunks)
    bounds = np.linspace(0, len(frames), n_chunks + 1).round().astype(int)
    return [frames[bounds[i]:bounds[i + 1]] for i in range(n_chunks)
            if bounds[i + 1] - bounds[i] >= SEQ_LEN]  # descarta pedacos curtos demais


def list_sequences() -> List[SequenceSample]:
    """
    Varre o dataset e devolve as amostras de sequencia.

    Sequencias longas sao divididas em sub-sequencias (chunks) de ate SEQ_CHUNK
    frames, cada uma virando uma SequenceSample independente. O `group` guarda o
    diretorio de origem para a validacao cruzada agrupada.
    """
    samples = []
    for cls_idx, cls in enumerate(CLASSES):
        cls_dir = SEQUENCES_DIR / cls
        if not cls_dir.exists():
            continue
        for seq_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            frames = sorted(
                (p for p in seq_dir.iterdir() if p.suffix.lower() in IMG_EXTS),
                key=_natural_key,
            )
            if not frames:
                continue
            for chunk_frames in _chunk(frames, SEQ_CHUNK):
                windows = _sample_windows(len(chunk_frames))
                samples.append(SequenceSample(chunk_frames, windows, cls_idx, seq_dir.name))
    return samples


def split_sequences():
    """
    Divide as sequencias em (treino, val), estratificado por classe e AGRUPADO
    por video de origem (todas as sub-sequencias de um mesmo video ficam do mesmo
    lado, evitando vazamento de frames quase identicos entre treino e validacao).
    """
    rng = np.random.default_rng(SEED)
    samples = list_sequences()
    if not samples:
        raise RuntimeError(
            f"Nenhuma sequencia encontrada em {SEQUENCES_DIR}. "
            "Rode extract_sequences.py ou verifique a estrutura de pastas."
        )

    train, val = [], []
    for cls_idx in range(len(CLASSES)):
        groups = sorted({s.group for s in samples if s.label == cls_idx})
        rng.shuffle(groups)
        n_val = max(1, int(round(len(groups) * VAL_SPLIT))) if len(groups) > 1 else 0
        val_groups = set(groups[:n_val])
        for s in samples:
            if s.label != cls_idx:
                continue
            (val if s.group in val_groups else train).append(s)

    print("Resumo do dataset:")
    for cls_idx, cls in enumerate(CLASSES):
        n_tr = sum(len(s.windows) for s in train if s.label == cls_idx)
        n_vl = sum(len(s.windows) for s in val if s.label == cls_idx)
        s_tr = sum(1 for s in train if s.label == cls_idx)
        s_vl = sum(1 for s in val if s.label == cls_idx)
        print(f"  {cls:8s}: treino {s_tr} seq / {n_tr} janelas | val {s_vl} seq / {n_vl} janelas")

    return train, val


def load_frame(path):
    """Le um frame como RGB float32 no tamanho IMG_SIZE (pixels em [0,255])."""
    img = cv2.imread(str(path))
    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img.astype(np.float32)


if __name__ == "__main__":
    split_sequences()
