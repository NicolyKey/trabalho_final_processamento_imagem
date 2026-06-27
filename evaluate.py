"""
Uso:
    python evaluate.py            # 5 folds (padrao)
    python evaluate.py 4          # k folds
"""
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold

from config import CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED
from dataset import list_sequences, load_frame
from motion_features import sequence_motion
from model import build_feature_extractor, build_temporal_head

tf.keras.utils.set_random_seed(SEED)


def per_sequence_features(samples, feature_extractor):
    out = []
    for s in samples:
        frames = np.stack([load_frame(p) for p in s.frames], axis=0)
        feats = feature_extractor.predict(frames, batch_size=32, verbose=0)
        motion = sequence_motion(s.frames)
        Xa = np.stack([feats[w] for w in s.windows], axis=0).astype(np.float32)
        Xm = np.stack([motion[w] for w in s.windows], axis=0).astype(np.float32)
        out.append((Xa, Xm, s.label))
    return out


def train_head(Xa, Xm, y, class_weight, n_classes):
    head = build_temporal_head(n_classes=n_classes)
    head.get_layer("motion_norm").adapt(Xm)
    head.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                 loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    head.fit([Xa, Xm], y, epochs=EPOCHS, batch_size=BATCH_SIZE,
             class_weight=class_weight, verbose=0,
             callbacks=[tf.keras.callbacks.EarlyStopping(
                 monitor="loss", patience=8, restore_best_weights=True)])
    return head


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    samples = list_sequences()
    labels = np.array([s.label for s in samples])
    groups = np.array([s.group for s in samples])
    n_classes = len(CLASSES)
    n_groups_min = min(len({s.group for s in samples if s.label == c})
                       for c in range(n_classes))
    print(f"{len(samples)} sub-sequencias de {len(set(groups))} videos | "
          f"classes={CLASSES} | validacao cruzada AGRUPADA por video\n")

    print("Extraindo features (uma vez)...")
    fe = build_feature_extractor()
    seq_feats = per_sequence_features(samples, fe)

    # k limitado pelo menor numero de VIDEOS por classe (folds agrupados).
    k = min(k, int(n_groups_min))
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=SEED)

    confusion = np.zeros((n_classes, n_classes), dtype=int)
    fold_acc = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(samples, labels, groups), 1):
        Xa_tr = np.concatenate([seq_feats[i][0] for i in tr_idx], axis=0)
        Xm_tr = np.concatenate([seq_feats[i][1] for i in tr_idx], axis=0)
        ytr = np.concatenate([np.full(len(seq_feats[i][0]), seq_feats[i][2]) for i in tr_idx])

        present = np.unique(ytr)
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight("balanced", classes=present, y=ytr)
        cw = {int(c): float(wi) for c, wi in zip(present, w)}

        head = train_head(Xa_tr, Xm_tr, ytr, cw, n_classes)

        correct = 0
        errors  = []
        for i in te_idx:
            Xa, Xm, true = seq_feats[i]
            probs = head.predict([Xa, Xm], verbose=0).mean(axis=0)
            pred  = int(np.argmax(probs))
            confusion[true, pred] += 1
            correct += int(pred == true)
            if pred != true:
                errors.append(
                    f"{samples[i].group:35s}  "
                    f"verdadeiro={CLASSES[true]:6s}  "
                    f"previsto={CLASSES[pred]:6s}  "
                    f"conf={float(probs[pred]):.2f}"
                )
        acc = correct / len(te_idx)
        fold_acc.append(acc)
        print(f"  fold {fold}: {len(tr_idx)} treino / {len(te_idx)} teste -> acc={acc:.3f}")
        if errors:
            for e in errors:
                print(f"    ✗ {e}")
        else:
            print(f"    ✓ sem erros")
            
    print(f"\nAcuracia media por sequencia (CV): {np.mean(fold_acc):.3f} "
          f"+/- {np.std(fold_acc):.3f}")
    print("\nMatriz de confusao (linhas=verdadeiro, colunas=previsto):")
    header = "          " + "  ".join(f"{c:>8s}" for c in CLASSES)
    print(header)
    for i, c in enumerate(CLASSES):
        row = "  ".join(f"{confusion[i, j]:8d}" for j in range(n_classes))
        print(f"  {c:>7s}  {row}")
    total = confusion.sum()
    print(f"\nAcuracia global (sequencias): {np.trace(confusion) / total:.3f} "
          f"({np.trace(confusion)}/{total})")


if __name__ == "__main__":
    main()
