# Classificação de Manobra de Bicicleta (360) — OpenCV + YOLO + TensorFlow

Identifica se uma bicicleta está executando uma manobra de **360** ou apenas
**andando normalmente**, a partir de uma sequência de frames de vídeo.

A classificação de uma manobra não pode ser feita olhando um único frame — ela
depende do **movimento ao longo do tempo** (a rotação). Por isso a abordagem é de
**classificação de sequência**: vários frames consecutivos da bicicleta são
analisados em conjunto, combinando **aparência** (CNN) e **movimento** (fluxo óptico).

## Pipeline

```
            ┌─────────┐   ┌──────────────┐   ┌───────────────────────────────┐   ┌──────────┐
  vídeo ──► │ OpenCV  ├─► │ YOLOv8       ├─► │ CNN-LSTM (TensorFlow)         ├─► │ 360 /    │
  (.mp4)    │ (frames)│   │ detecta+crop │   │ aparência (MobileNetV2)       │   │ normal   │
            └─────────┘   │ da bicicleta │   │      +                        │   └──────────┘
                          └──────────────┘   │ movimento (fluxo óptico/OpenCV)│
                                             │      -> LSTM -> softmax        │
                                             └───────────────────────────────┘
```

1. **OpenCV** lê o vídeo quadro a quadro.
2. **YOLOv8** (ultralytics) detecta a bicicleta (classe COCO `bicycle`, com
   fallback para `person`) e recorta a região.
3. Os recortes são agrupados em janelas que cobrem a **manobra inteira**
   (`SEQ_LEN` frames subamostrados ao longo de `SPAN` frames de origem).
4. **CNN-LSTM** (TensorFlow/Keras), com **duas entradas por frame**:
   - **Aparência** — MobileNetV2 pré-treinada (congelada, via `TimeDistributed`)
     extrai 1280 características espaciais por frame;
   - **Movimento** — fluxo óptico denso (Farneback, OpenCV) resumido em
     descritores de rotação (*curl*), divergência e magnitude;
   - as duas streams são concatenadas e processadas por uma **LSTM**, seguida de
     uma camada densa que decide `360` vs `normal`.

## Estrutura do projeto

```
config.py             parâmetros centrais (SEQ_LEN, SPAN, FRAME_STEP, caminhos)
extract_sequences.py  gera sequências a partir de videos/ usando YOLO+OpenCV
motion_features.py    descritores de movimento via fluxo óptico (OpenCV)
dataset.py            carrega sequences_dataset/, faz janelas e split por vídeo
model.py              arquitetura CNN-LSTM de 2 entradas (aparência + movimento)
train.py              treina e salva o modelo + classes + gráfico
predict.py            inferência em vídeo (YOLO + CNN-LSTM + overlay)
evaluate.py           validação cruzada honesta (agrupada por vídeo)
diagnose.py           diagnóstico: separabilidade por aparência
diagnose_motion.py    diagnóstico: separabilidade por movimento
diagnose_bbox.py      diagnóstico: separabilidade por trajetória da bbox

sequences_dataset/    dataset de sequências (360/ e normal/)
videos/               vídeos brutos (fonte para gerar mais sequências)
models/               saídas do treino (modelo .keras, classes.txt, gráfico)
output/               vídeos anotados da inferência
```

## Como usar

### 1. Instalar dependências

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. (Opcional) Expandir o dataset a partir dos vídeos

Usa YOLO para recortar a bicicleta de todos os vídeos de `videos/` e criar novas
sequências. A classe é inferida pelo nome do arquivo (`360` no nome → classe
`360`, senão `normal`).

```bash
python extract_sequences.py
```

### 3. Treinar o modelo

```bash
python train.py
```

Gera `models/trick_classifier.keras`, `models/classes.txt` e
`models/training_history.png`. (Use `--no-cache` para recomputar as features.)

### 4. Avaliar de forma honesta

```bash
python evaluate.py 5     # validação cruzada agrupada por vídeo (5 folds)
```

### 5. Inferência em um vídeo

```bash
python predict.py videos/360_estavel.mp4 --out output/resultado.mp4
```

## Parâmetros principais (`config.py`)

| Parâmetro       | Padrão | Descrição                                        |
|-----------------|--------|--------------------------------------------------|
| `SEQ_LEN`       | 16     | nº de frames por amostra (entrada da LSTM)       |
| `FRAME_STEP`    | 3      | subamostragem temporal dentro de uma amostra     |
| `SPAN`          | 48     | frames de origem cobertos por amostra (=16×3)    |
| `IMG_SIZE`      | 128    | resolução de cada frame recortado                |
| `SEQ_CHUNK`     | 90     | divide sequências longas em sub-sequências       |
| `YOLO_CONF`     | 0.25   | confiança mínima da detecção YOLO                |

## Resultados e limitações (importante)

O dataset disponível tem **15 vídeos de `360`** e apenas **5 de `normal`**. Sob
**validação cruzada agrupada por vídeo** (a forma honesta de medir — sequências
de um mesmo vídeo nunca aparecem em treino e teste ao mesmo tempo), a acurácia
média por sequência fica em torno de **0,58–0,65**.

A matriz de confusão revela o ponto central:

```
              previsto:  360   normal
verdadeiro 360:           ~27     ~5     -> reconhece 360 bem
verdadeiro normal:        ~15      0     -> falha em generalizar "normal"
```

Investigamos a fundo a causa (ver os scripts `diagnose*.py`):

- **Aparência (CNN)**, **movimento (fluxo óptico)** e **trajetória da bbox**,
  isoladamente ou combinados, batem todos na mesma parede.
- O modelo reconhece `360` (classe com 15 vídeos), mas **não generaliza `normal`**
  a partir de apenas 3–4 vídeos de treino para um vídeo novo — recai na classe
  majoritária.

**Conclusão:** o gargalo é a **quantidade e o equilíbrio de vídeos `normal`**,
não a arquitetura. A pipeline (YOLO → CNN-LSTM com aparência + movimento) está
completa e correta.

### Como melhorar a acurácia (em ordem de impacto)

1. **Gravar mais vídeos `normal`** (10+), de ângulos/iluminações variados — é de
   longe o fator mais importante para equilibrar as classes.
2. Gravar mais variações de `360` (frente, lado, no ar, lento/rápido).
3. Após ter dados balanceados, opcionalmente descongelar as últimas camadas da
   MobileNetV2 (fine-tuning) e aumentar a capacidade da LSTM.
