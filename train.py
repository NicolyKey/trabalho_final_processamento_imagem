"""
Treino do classificador de manobras CNN-LSTM.

Estrategia (rapida em CPU): como o backbone CNN esta congelado, as features de
cada frame sao fixas. Entao:
  1. extraimos as features de cada frame UMA vez (MobileNetV2);
  2. montamos as janelas de features (SEQ_LEN, 1280);
  3. treinamos so a cabeca temporal (LSTM) sobre essas features;
  4. remontamos o modelo completo end-to-end e salvamos para a inferencia.

Salva:
    models/trick_classifier.keras   -> modelo completo (frames -> classe)
    models/classes.txt              -> nomes das classes (ordem dos indices)
    models/training_history.png     -> curvas de loss/accuracy

Uso:
    python train.py
"""
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import (
    CLASSES, FRAME_STEP, IMG_SIZE, MODEL_PATH, CLASSES_PATH, HISTORY_PLOT, MODELS_DIR,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, SEED, SEQ_LEN, SPAN
)

import hashlib, json
cfg = {"SEQ_LEN": SEQ_LEN, "FRAME_STEP": FRAME_STEP, "IMG_SIZE": IMG_SIZE, "SPAN": SPAN}
cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
FEATURES_CACHE = MODELS_DIR / f"features_cache_{cfg_hash}.npz"
from dataset import split_sequences, load_frame
from motion_features import sequence_motion, MOTION_DIM
from model import (
    build_feature_extractor, build_temporal_head, build_full_model, FEATURE_DIM,
)

tf.keras.utils.set_random_seed(SEED)


def build_feature_windows(samples, feature_extractor):
    """
    Para cada sequencia, extrai as features de APARENCIA (CNN) e de MOVIMENTO
    (fluxo optico) de seus frames uma unica vez e monta as janelas.

    Devolve (X_app, X_mot, y):
      X_app: (N, SEQ_LEN, FEATURE_DIM)
      X_mot: (N, SEQ_LEN, MOTION_DIM)
    """
    Xa, Xm, y = [], [], []
    for s in samples:
        frames = np.stack([load_frame(p) for p in s.frames], axis=0)        # (n, IMG, IMG, 3)
        feats = feature_extractor.predict(frames, batch_size=32, verbose=0)  # (n, 1280)
        motion = sequence_motion(s.frames)                                   # (n, MOTION_DIM)
        for win in s.windows:
            Xa.append(feats[win])
            Xm.append(motion[win])
            y.append(s.label)
    if not Xa:
        return (np.empty((0, SEQ_LEN, FEATURE_DIM), np.float32),
                np.empty((0, SEQ_LEN, MOTION_DIM), np.float32),
                np.empty((0,), np.int64))
    return (np.asarray(Xa, dtype=np.float32),
            np.asarray(Xm, dtype=np.float32),
            np.asarray(y, dtype=np.int64))


def plot_history(history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib indisponivel, pulando grafico.")
        return
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h.get("loss", []), label="treino")
    if "val_loss" in h:
        axes[0].plot(h["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoca"); axes[0].legend()
    axes[1].plot(h.get("accuracy", []), label="treino")
    if "val_accuracy" in h:
        axes[1].plot(h["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("epoca"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(HISTORY_PLOT, dpi=120)
    print(f"Grafico salvo em {HISTORY_PLOT}")


def main():
    import sys
    use_cache = "--no-cache" not in sys.argv

    train_seqs, val_seqs = split_sequences()
    feature_extractor = build_feature_extractor()

    if use_cache and FEATURES_CACHE.exists():
        print(f"\nCarregando features do cache: {FEATURES_CACHE}")
        d = np.load(FEATURES_CACHE)
        Xa_tr, Xm_tr, y_train = d["Xa_tr"], d["Xm_tr"], d["ytr"]
        Xa_vl, Xm_vl, y_val = d["Xa_vl"], d["Xm_vl"], d["yvl"]
    else:
        print("\nExtraindo features de aparencia (CNN) e movimento (fluxo optico)...")
        Xa_tr, Xm_tr, y_train = build_feature_windows(train_seqs, feature_extractor)
        Xa_vl, Xm_vl, y_val = build_feature_windows(val_seqs, feature_extractor)
        np.savez(FEATURES_CACHE, Xa_tr=Xa_tr, Xm_tr=Xm_tr, ytr=y_train,
                 Xa_vl=Xa_vl, Xm_vl=Xm_vl, yvl=y_val)
        print(f"Features cacheadas em {FEATURES_CACHE} (use --no-cache para recomputar)")
    print(f"  aparencia treino={Xa_tr.shape}  movimento treino={Xm_tr.shape}  val={Xa_vl.shape}")

    has_val = len(Xa_vl) > 0

    # Pesos de classe para compensar desbalanceamento.
    present = np.unique(y_train)
    if len(present) > 1:
        weights = compute_class_weight("balanced", classes=present, y=y_train)
        class_weight = {int(c): float(w) for c, w in zip(present, weights)}
    else:
        class_weight = None
    print(f"Pesos de classe: {class_weight}")

    # --- Treina a cabeca temporal (aparencia + movimento) ---
    head = build_temporal_head(n_classes=len(CLASSES))
    # Adapta a normalizacao do movimento com as estatisticas do TREINO.
    head.get_layer("motion_norm").adapt(Xm_tr)
    head.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    head.summary()

    monitor = "val_accuracy" if has_val else "accuracy"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor, mode="max", patience=12,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
        ),
    ]

    Xa_flip = Xa_tr[:, ::-1, :]
    Xm_flip = Xm_tr[:, ::-1, :]
    Xm_flip[:, :, 3] *= -1  # inverte sinal do curl

    Xa_tr = np.concatenate([Xa_tr, Xa_flip], axis=0)
    Xm_tr = np.concatenate([Xm_tr, Xm_flip], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)

    history = head.fit(
        [Xa_tr, Xm_tr], y_train,
        validation_data=([Xa_vl, Xm_vl], y_val) if has_val else None,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    if has_val:
        loss, acc = head.evaluate([Xa_vl, Xm_vl], y_val, verbose=0)
        print(f"\nValidacao final -> loss={loss:.4f} accuracy={acc:.4f}")

    # --- Remonta o modelo completo (reaproveita os MESMOS objetos -> mantem pesos) ---
    full_model = build_full_model(feature_extractor, head)
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    full_model.save(MODEL_PATH)
    CLASSES_PATH.write_text("\n".join(CLASSES) + "\n")
    print(f"\nModelo completo salvo em {MODEL_PATH}")
    print(f"Classes salvas em {CLASSES_PATH}")

    plot_history(history)


if __name__ == "__main__":
    main()
