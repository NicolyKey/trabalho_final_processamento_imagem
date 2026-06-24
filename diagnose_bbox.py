"""
Diagnostico: a TRAJETORIA da bounding box (geometria ao longo do tempo) separa
360 de normal? Roda YOLO sobre cada video, monta a serie temporal da bbox
(posicao, tamanho, proporcao) normalizada pelo quadro e testa a separabilidade.
"""
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import VIDEOS_DIR, CLASSES, SEED, YOLO_WEIGHTS, YOLO_CONF
from extract_sequences import best_detection, class_for_video


def bbox_trajectory(model, video_path):
    cap = cv2.VideoCapture(str(video_path))
    W = cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
    H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=YOLO_CONF, verbose=False)
        box = best_detection(res[0])
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w, h = (x2 - x1) / W, (y2 - y1) / H
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        aspect = (x2 - x1) / max(1.0, (y2 - y1))
        rows.append([cx, cy, w, h, aspect])
    cap.release()
    return np.asarray(rows, dtype=np.float32)


def traj_features(traj):
    """Estatisticas da trajetoria: variabilidade de posicao/tamanho/proporcao."""
    if len(traj) < 3:
        return np.zeros(15, np.float32)
    cx, cy, w, h, aspect = traj.T
    d = np.diff(traj, axis=0)          # variacoes frame a frame
    feats = [
        cy.std(), cx.std(), h.std(), w.std(), aspect.std(),
        aspect.max() - aspect.min(),   # amplitude da proporcao (giro muda muito)
        cy.max() - cy.min(),           # amplitude vertical (pulo/ar)
        np.abs(d[:, 1]).mean(),        # movimento vertical medio
        np.abs(d[:, 0]).mean(),        # movimento horizontal medio
        np.abs(d[:, 4]).mean(),        # variacao media da proporcao
        np.abs(d[:, 4]).std(),
        h.mean(), w.mean(), aspect.mean(), aspect.std() / max(1e-6, aspect.mean()),
    ]
    return np.asarray(feats, np.float32)


def main():
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)

    X, y, g = [], [], []
    for v in sorted(VIDEOS_DIR.glob("*.mp4")):
        traj = bbox_trajectory(model, v)
        X.append(traj_features(traj))
        y.append(CLASSES.index(class_for_video(v.name)))
        g.append(v.stem)
        print(f"  {v.name:28s} classe={class_for_video(v.name):6s} frames={len(traj)}")
    X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
    print("\nX:", X.shape, "| classes:", np.bincount(y))

    k = min(5, min((y == c).sum() for c in range(len(CLASSES))))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
    conf = np.zeros((len(CLASSES), len(CLASSES)), int); accs = []
    for tr, te in skf.split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
        for t, p in zip(y[te], pred):
            conf[t, p] += 1
        accs.append((pred == y[te]).mean())

    print(f"\nLogReg sobre TRAJETORIA da bbox (CV agrupada por video): "
          f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")
    print("Confusao (linha=verdadeiro, col=previsto):", CLASSES)
    print(conf)
    print(f"Acuracia global: {np.trace(conf)/conf.sum():.3f} ({np.trace(conf)}/{conf.sum()})")


if __name__ == "__main__":
    main()
