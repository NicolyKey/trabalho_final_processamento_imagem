import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from config import CLASSES, SEED
from dataset import list_sequences, load_frame
from model import build_feature_extractor

samples = list_sequences()
fe = build_feature_extractor()

X, y, g = [], [], []
for s in samples:
    frames = np.stack([load_frame(p) for p in s.frames], axis=0)
    feats = fe.predict(frames, batch_size=32, verbose=0)         # (n, 1280)
    # Estatisticas temporais: media e desvio-padrao ao longo do tempo.
    stat = np.concatenate([feats.mean(0), feats.std(0)])          # (2560,)
    X.append(stat); y.append(s.label); g.append(s.group)
X = np.asarray(X); y = np.asarray(y); g = np.asarray(g)
print("X:", X.shape, "| classes:", np.bincount(y))

# Sinal simples: magnitude media da variacao temporal das features por sequencia.
var_signal = X[:, 1280:].mean(1)
for c, name in enumerate(CLASSES):
    v = var_signal[y == c]
    print(f"  variacao temporal media [{name}]: {v.mean():.3f} +/- {v.std():.3f}")

k = min(5, min((y == c).sum() for c in range(len(CLASSES))))
skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)
conf = np.zeros((len(CLASSES), len(CLASSES)), int)
accs = []
for tr, te in skf.split(X, y, g):
    sc = StandardScaler().fit(X[tr])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    for t, p in zip(y[te], pred):
        conf[t, p] += 1
    accs.append((pred == y[te]).mean())

print(f"\nLogReg sobre estatisticas temporais (CV agrupada): "
      f"acc={np.mean(accs):.3f} +/- {np.std(accs):.3f}")
print("Confusao (linha=verdadeiro, col=previsto):", CLASSES)
print(conf)
print(f"Acuracia global: {np.trace(conf)/conf.sum():.3f} ({np.trace(conf)}/{conf.sum()})")
