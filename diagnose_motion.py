import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import CLASSES, SEED
from dataset import list_sequences
from motion_features import sequence_motion, MOTION_NAMES

samples = list_sequences()
X, y, g = [], [], []
for s in samples:
    m = sequence_motion(s.frames)              # (n, MOTION_DIM)
    # Estatisticas temporais do movimento: media e desvio ao longo do tempo.
    stat = np.concatenate([m.mean(0), m.std(0)])
    X.append(stat); y.append(s.label); g.append(s.group)
X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
print("X:", X.shape, "| classes:", np.bincount(y))

# Compara o sinal de rotacao (curl) entre as classes.
curl_idx = MOTION_NAMES.index("curl_abs_mean")
for c, name in enumerate(CLASSES):
    v = X[y == c, curl_idx]
    print(f"  curl_abs_mean [{name:6s}]: {v.mean():.4f} +/- {v.std():.4f}")

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

print(f"\nLogReg sobre features de MOVIMENTO (CV agrupada): "
      f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")
print("Confusao (linha=verdadeiro, col=previsto):", CLASSES)
print(conf)
print(f"Acuracia global: {np.trace(conf)/conf.sum():.3f} ({np.trace(conf)}/{conf.sum()})")
