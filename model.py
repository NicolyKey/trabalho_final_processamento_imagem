from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2

from config import SEQ_LEN, IMG_SIZE, CHANNELS, CLASSES
from motion_features import MOTION_DIM

FEATURE_DIM = 1280  # dimensao de saida do MobileNetV2 com pooling='avg'


def build_feature_extractor(img_size=IMG_SIZE, channels=CHANNELS):
    backbone = MobileNetV2(
        input_shape=(img_size, img_size, channels),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    backbone.trainable = False

    inp = layers.Input(shape=(img_size, img_size, channels))
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0)(inp)
    out = backbone(x, training=False)
    return models.Model(inp, out, name="mobilenetv2_features")


def build_temporal_head(seq_len=SEQ_LEN, feat_dim=FEATURE_DIM,
                        motion_dim=MOTION_DIM, n_classes=len(CLASSES)):
    reg = regularizers.l2(1e-4)
    app_in = layers.Input(shape=(seq_len, feat_dim), name="appearance")
    mot_in = layers.Input(shape=(seq_len, motion_dim), name="motion")

    a = layers.Dropout(0.5)(app_in)
    a = layers.TimeDistributed(
        layers.Dense(64, activation="relu", kernel_regularizer=reg))(a)
    # Normaliza os descritores de movimento (adaptado no treino; ver train.py).
    m = layers.Normalization(axis=-1, name="motion_norm")(mot_in)
    m = layers.TimeDistributed(
        layers.Dense(16, activation="relu", kernel_regularizer=reg))(m)

    x = layers.Concatenate()([a, m])
    x = layers.Dropout(0.5)(x)
    x = layers.LSTM(32, return_sequences=False,
                    dropout=0.3, recurrent_dropout=0.3,
                    kernel_regularizer=reg, recurrent_regularizer=reg)(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg)(x)
    return models.Model([app_in, mot_in], out, name="temporal_head")


def build_full_model(feature_extractor, temporal_head,
                     seq_len=SEQ_LEN, img_size=IMG_SIZE, channels=CHANNELS,
                     motion_dim=MOTION_DIM):
    frames_in = layers.Input(shape=(seq_len, img_size, img_size, channels),
                             name="frames")
    motion_in = layers.Input(shape=(seq_len, motion_dim), name="motion")
    app = layers.TimeDistributed(feature_extractor)(frames_in)  # (seq_len, FEATURE_DIM)
    out = temporal_head([app, motion_in])
    return models.Model([frames_in, motion_in], out, name="bike_trick_cnn_lstm")


if __name__ == "__main__":
    fe = build_feature_extractor()
    head = build_temporal_head()
    full = build_full_model(fe, head)
    full.summary()
