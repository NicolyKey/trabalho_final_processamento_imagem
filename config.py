"""
Configuracao central do projeto de classificacao de manobras de bicicleta.

Pipeline:
    OpenCV  -> leitura de video / pre-processamento
    YOLO    -> deteccao e recorte da bicicleta em cada frame
    TensorFlow (CNN-LSTM) -> classifica a sequencia de frames como "360" ou "normal"
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SEQUENCES_DIR = ROOT / "sequences_dataset"   # dataset de sequencias (entrada do treino)
VIDEOS_DIR = ROOT / "videos"                 # videos brutos (para gerar mais sequencias)
MODELS_DIR = ROOT / "models"                 # saida: modelo treinado + artefatos
OUTPUT_DIR = ROOT / "output"                 # saida: videos/resultados da inferencia

MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODELS_DIR / "trick_classifier.keras"
CLASSES_PATH = MODELS_DIR / "classes.txt"
HISTORY_PLOT = MODELS_DIR / "training_history.png"

# ----------------------------------------------------------------------------
# Parametros da sequencia / imagem
# ----------------------------------------------------------------------------
# Cada amostra para o modelo e uma sequencia de SEQ_LEN frames recortados da bike.
SEQ_LEN = 16          # numero de frames por amostra (entrada da LSTM)
IMG_SIZE = 128        # resolucao (largura=altura) de cada frame recortado
CHANNELS = 3

# Subamostragem temporal: cada amostra cobre SEQ_LEN * FRAME_STEP frames de origem,
# pegando 1 a cada FRAME_STEP. Isso faz cada amostra abranger a MANOBRA INTEIRA
# (um 360 leva dezenas de frames), e nao apenas um trecho curto que se parece com
# pedalada normal. O mesmo span e usado no treino e na inferencia (predict.py).
FRAME_STEP = 3        # passo de subamostragem dentro de uma amostra
SPAN = SEQ_LEN * FRAME_STEP  # nº de frames de origem cobertos por amostra (=48)

# Janela deslizante entre amostras consecutivas (data augmentation temporal).
WINDOW_STRIDE = 12    # passo, em frames de origem, entre amostras

# Sequencias muito longas sao divididas em sub-sequencias de ate SEQ_CHUNK frames.
# Isso gera mais amostras independentes e equilibra melhor as classes (os videos
# "normal" sao longos). As sub-sequencias guardam o "grupo" (video de origem) para
# a validacao cruzada agrupada nao vazar frames do mesmo video entre treino/teste.
SEQ_CHUNK = 90

# ----------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------
# Determinadas pelas subpastas de sequences_dataset/. Mantido explicito para a
# inferencia funcionar mesmo sem o dataset presente.
CLASSES = ["360", "normal"]

# ----------------------------------------------------------------------------
# YOLO / deteccao
# ----------------------------------------------------------------------------
YOLO_WEIGHTS = "yolov8n.pt"   # baixado automaticamente pela ultralytics se ausente
YOLO_CONF = 0.25              # confianca minima da deteccao
# Classes COCO de interesse: 1 = bicycle, 0 = person (ciclista, usado como fallback)
COCO_BICYCLE = 1
COCO_PERSON = 0
CROP_MARGIN = 0.15            # margem extra ao redor da bbox ao recortar (15%)

# ----------------------------------------------------------------------------
# Treino
# ----------------------------------------------------------------------------
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2       # fracao das sequencias reservada para validacao
SEED = 42
