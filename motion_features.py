import numpy as np
import cv2

from config import IMG_SIZE, FRAME_STEP

FLOW_SIZE = 96            # resolucao em que o fluxo e calculado (mais barato)
MOTION_DIM = 7           # dimensao do descritor de movimento por frame

# Nomes dos componentes do descritor (para documentacao/diagnostico).
MOTION_NAMES = [
    "mag_mean", "mag_std", "curl_abs_mean", "curl_mean",
    "div_abs_mean", "div_mean", "mag_p90",
]


def _gray(path_or_img):
    if isinstance(path_or_img, np.ndarray):
        img = path_or_img
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = cv2.imread(str(path_or_img), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((FLOW_SIZE, FLOW_SIZE), np.uint8)
    return cv2.resize(img, (FLOW_SIZE, FLOW_SIZE))


def flow_descriptor(prev_gray, cur_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.sqrt(fx * fx + fy * fy)

    # Gradientes para curl (rotacao) e divergencia (expansao/translacao).
    dfx_dy, dfx_dx = np.gradient(fx)
    dfy_dy, dfy_dx = np.gradient(fy)
    curl = dfy_dx - dfx_dy          # vorticidade -> rotacao
    div = dfx_dx + dfy_dy           # divergencia

    return np.array([
        mag.mean(),
        mag.std(),
        np.abs(curl).mean(),
        curl.mean(),
        np.abs(div).mean(),
        div.mean(),
        np.percentile(mag, 90),
    ], dtype=np.float32)


def sequence_motion(frame_paths):
    grays = [_gray(p) for p in frame_paths]
    n = len(grays)
    out = np.zeros((n, MOTION_DIM), dtype=np.float32)
    for i in range(FRAME_STEP, n):
        out[i] = flow_descriptor(grays[i - FRAME_STEP], grays[i])
    # Preenche o inicio repetindo o primeiro descritor valido.
    if n > FRAME_STEP:
        out[:FRAME_STEP] = out[FRAME_STEP]
    return out
