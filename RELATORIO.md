# Detecção Automática de Manobras "360" em Vídeos de Ciclismo

### Relatório Técnico-Acadêmico — Trabalho Final de Processamento de Imagens

---

## Sumário

1. [Resumo](#1-resumo)
2. [Objetivo e definição do problema](#2-objetivo-e-definição-do-problema)
3. [Visão geral do método (pipeline)](#3-visão-geral-do-método-pipeline)
4. [Bibliotecas utilizadas](#4-bibliotecas-utilizadas)
5. [Etapa 1 — Detecção e rastreamento da bicicleta (YOLOv8 + ByteTrack)](#5-etapa-1--detecção-e-rastreamento-da-bicicleta-yolov8--bytetrack)
6. [Etapa 2 — Interpolação de detecções perdidas](#6-etapa-2--interpolação-de-detecções-perdidas)
7. [Etapa 3 — Recorte e janela temporal deslizante](#7-etapa-3--recorte-e-janela-temporal-deslizante)
8. [Etapa 4 — Classificação da manobra](#8-etapa-4--classificação-da-manobra)
9. [Etapa 5 — Anotação e geração do vídeo de saída](#9-etapa-5--anotação-e-geração-do-vídeo-de-saída)
10. [Preparação do dataset](#10-preparação-do-dataset)
11. [Treinamento dos modelos](#11-treinamento-dos-modelos)
12. [Tutorial: como reproduzir o resultado do zero](#12-tutorial-como-reproduzir-o-resultado-do-zero)
13. [Parâmetros e ajustes finos](#13-parâmetros-e-ajustes-finos)
14. [Limitações e trabalhos futuros](#14-limitações-e-trabalhos-futuros)
15. [Conclusão](#15-conclusão)
16. [Referências](#16-referências)

---

## 1. Resumo

Este trabalho descreve um sistema de **visão computacional** capaz de localizar uma bicicleta em um vídeo e identificar automaticamente o instante em que o ciclista executa a manobra **"360"** (giro completo de 360° em torno do eixo vertical). O sistema combina três famílias de técnicas:

1. **Detecção de objetos com aprendizado profundo** (YOLOv8) para localizar a bicicleta em cada quadro;
2. **Rastreamento multiquadro** (ByteTrack) com **interpolação de movimento** para manter a identidade da bicicleta mesmo quando a detecção falha em quadros isolados;
3. **Classificação temporal da manobra**, feita por duas vias complementares: uma rede convolucional por *transfer learning* (MobileNetV2) combinada com **análise clássica de movimento** (diferença entre quadros e variância de histograma), e uma rede **CNN + LSTM / Conv3D** que aprende diretamente o padrão espaço-temporal do giro.

O resultado é um vídeo anotado em que a bicicleta é marcada quadro a quadro e a legenda *"MANOBRA 360 DETECTADA"* aparece durante a execução do giro.

---

## 2. Objetivo e definição do problema

**Objetivo geral:** dado um vídeo `MP4` contendo um ciclista, produzir um novo vídeo no qual:

- a bicicleta seja detectada e contornada por uma *bounding box* em todos os quadros;
- os trechos em que ocorre a manobra "360" sejam destacados em vermelho e rotulados com a confiança da detecção.

**Por que o problema é difícil?** Um classificador ingênuo que olha um único quadro não consegue distinguir um "360" de uma bicicleta simplesmente apontada para o lado — **a informação que define a manobra está no tempo**, não em uma imagem isolada. O giro só é reconhecível observando a *sequência* de quadros: a aparência da bicicleta muda rapidamente (frente → lateral → traseira → lateral → frente). Por isso o sistema é construído em torno de uma **análise temporal**, e não de uma classificação de imagem única.

Formalmente, o problema é de **detecção de ação em vídeo** (*temporal action detection*): localizar no eixo do tempo o intervalo `[t_início, t_fim]` em que a ação "360" ocorre.

---

## 3. Visão geral do método (pipeline)

O processamento de cada quadro segue o fluxo abaixo:

```
┌─────────────┐   ┌──────────────────────┐   ┌────────────────────┐
│  Vídeo .mp4 │──▶│  Leitura quadro a     │──▶│  YOLOv8 + ByteTrack │
│  (entrada)  │   │  quadro (OpenCV)      │   │  detecta a bike     │
└─────────────┘   └──────────────────────┘   └─────────┬──────────┘
                                                        │ bbox + id
                                                        ▼
                          ┌──────────────────────────────────────────┐
                          │  Interpolação de detecções perdidas       │
                          │  (preenche gaps de até 5 quadros)          │
                          └─────────────────────┬────────────────────┘
                                                │ recorte da bike
                                                ▼
                          ┌──────────────────────────────────────────┐
                          │  Buffer temporal por track_id             │
                          │  (janela deslizante de N quadros)         │
                          └─────────────────────┬────────────────────┘
                                                │ sequência de recortes
                                                ▼
                          ┌──────────────────────────────────────────┐
                          │  Classificação da manobra                 │
                          │  CNN (MobileNetV2) + análise de movimento  │
                          │  ── ou ── CNN+LSTM / Conv3D                │
                          └─────────────────────┬────────────────────┘
                                                │ "360" ou "normal" + confiança
                                                ▼
                          ┌──────────────────────────────────────────┐
                          │  Anotação do quadro + escrita no vídeo     │
                          │  (OpenCV VideoWriter)                      │
                          └─────────────────────┬────────────────────┘
                                                ▼
                                        ┌───────────────┐
                                        │ resultado.mp4 │
                                        └───────────────┘
```

O código está organizado em módulos que espelham essas etapas:

| Arquivo | Papel no pipeline |
|---|---|
| `main.py` | Orquestra todo o fluxo (`BikeManeuverDetector`) e a interface de linha de comando |
| `bike_detector.py` | Detecção (YOLO), rastreamento (ByteTrack) e interpolação de quadros |
| `trick_classifier.py` | Classificador por quadro (MobileNetV2) + análise de movimento clássica |
| `sequence_trick_classifier.py` | Classificador sequencial (CNN+LSTM e Conv3D) |
| `sequence_data_generator.py` | Gerador de lotes (*batches*) de sequências para treino |
| `train_model.py` / `train_sequence_model.py` | Scripts de treinamento dos dois classificadores |
| `prepare_dataset.py` | Verificação e organização do conjunto de dados |

---

## 4. Bibliotecas utilizadas

O projeto usa um conjunto enxuto de bibliotecas, cada uma com um papel bem definido. As versões abaixo foram verificadas no ambiente do projeto.

| Biblioteca | Versão (ambiente) | Função no projeto |
|---|---|---|
| **OpenCV** (`opencv-python`) | 4.13 | Leitura/escrita de vídeo, recorte, redimensionamento, conversão de cor, cálculo de histograma e desenho das anotações |
| **NumPy** | 2.5 | Representação das imagens como arrays e toda a aritmética vetorizada (diferenças, médias, interpolação) |
| **TensorFlow / Keras** | 2.21 | Construção, treinamento e inferência das redes neurais de classificação |
| **Ultralytics (YOLOv8)** | 8.4 | Detecção da bicicleta e rastreamento integrado (ByteTrack) |
| **Pillow** | — | Dependência de imagem usada internamente pelo Keras/Ultralytics |
| **Matplotlib** | — | Visualização opcional de métricas de treinamento |

> **`requirements.txt`**
> ```
> opencv-python>=4.8.0
> numpy>=1.24.0
> tensorflow>=2.13.0
> ultralytics>=8.0.0
> Pillow>=10.0.0
> matplotlib>=3.7.0
> ```

### 4.1. Por que cada biblioteca?

- **OpenCV** é a espinha dorsal de *I/O* e processamento de imagem. Toda a manipulação pixel-a-pixel (recortes, *resize*, escala de cinza, histogramas) e a montagem do vídeo final passam por ela. É escolhida por ser madura, rápida (núcleo em C++) e padrão de mercado em visão computacional.
- **NumPy** fornece o tipo de dado fundamental: uma imagem, no OpenCV, **já é** um `numpy.ndarray` de forma `(altura, largura, canais)`. Operações como "diferença média entre dois quadros" são escritas como aritmética de arrays, sem laços explícitos em Python.
- **Ultralytics/YOLOv8** entrega detecção *estado-da-arte* com uma API de uma linha, incluindo rastreamento. Evita treinar um detector de objetos do zero — algo inviável dentro do escopo do trabalho.
- **TensorFlow/Keras** é o *framework* de aprendizado profundo onde os classificadores são definidos, treinados e carregados. O uso de `keras.applications.MobileNetV2` permite *transfer learning* com pesos pré-treinados na ImageNet.

---

## 5. Etapa 1 — Detecção e rastreamento da bicicleta (YOLOv8 + ByteTrack)

A classe `BikeDetector` (`bike_detector.py`) encapsula a detecção. O coração é uma única chamada à API do YOLO em modo de **rastreamento**:

```python
results = self.model.track(
    frame,
    persist=True,          # mantém o estado do tracker entre quadros
    verbose=False,
    imgsz=1280,            # resolução de inferência (maior = mais sensível)
    conf=self.confidence_threshold,  # 0.4
    iou=0.5,               # limiar de IoU para supressão não-máxima
    tracker='bytetrack.yaml'
)
```

### 5.1. O que o YOLO faz

YOLO (*You Only Look Once*) é um detector de objetos de **estágio único**: em uma única passagem pela rede, ele prevê simultaneamente *onde* estão os objetos (coordenadas das caixas) e *o que* são (classe + confiança). O modelo usado, `yolov8m.pt`, foi pré-treinado no conjunto **COCO**, que inclui a classe `bicycle` (índice 1). O código filtra apenas essa classe:

```python
self.bike_class_id = 1
...
if class_id == self.bike_class_id and confidence >= self.confidence_threshold:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # canto superior-esq. e inferior-dir.
    track_id = int(box.id[0]) if box.id is not None else 0
```

Cada detecção devolve uma **bounding box** `[x1, y1, x2, y2]`, a **confiança** (probabilidade ∈ [0,1]) e um **`track_id`** atribuído pelo rastreador.

### 5.2. O papel do ByteTrack

Detectar não basta: precisamos saber que a bicicleta do quadro 100 é a *mesma* do quadro 101, para acumular sua sequência temporal. O **ByteTrack** resolve isso associando detecções ao longo do tempo e atribuindo um `track_id` estável. Ele é robusto porque também aproveita detecções de baixa confiança para manter o rastro, reduzindo trocas de identidade.

### 5.3. Decisões de engenharia importantes

- **Limiar de confiança baixo (`conf=0.4`)**: durante o giro, a bicicleta aparece borrada e em ângulos atípicos, derrubando a confiança da detecção. Um limiar menor captura mais quadros do momento crítico — exatamente o que queremos preservar. (Documentado em `MELHORIAS_DETECCAO.md`.)
- **Resolução alta (`imgsz=1280`)**: aumenta a sensibilidade a objetos pequenos/rápidos, ao custo de mais processamento.
- **Histórico de rastreamento**: para cada `track_id`, guarda-se uma lista das últimas detecções *reais* (até `max_history = 30`), usada na interpolação.

---

## 6. Etapa 2 — Interpolação de detecções perdidas

Mesmo com um limiar baixo, o YOLO **falha em quadros isolados** no meio do giro, criando "buracos" (*gaps*) na sequência. Uma sequência com lacunas prejudica tanto a visualização quanto o classificador temporal. A solução implementada é a **interpolação linear de movimento** em `_interpolate_missing_tracks()`.

### 6.1. Princípio

Quando um `track_id` que existia deixa de ser detectado por até `max_frames_missing = 5` quadros, o sistema **estima** onde a bicicleta deveria estar, projetando a trajetória a partir da **velocidade** das duas últimas detecções reais:

```python
# Velocidade (deslocamento por quadro) estimada das 2 últimas detecções reais
dt = (last_real['frame'] - prev_detection['frame']) + 1e-6
dx = (last_real['bbox'][0] - prev_detection['bbox'][0]) / dt
dy = (last_real['bbox'][1] - prev_detection['bbox'][1]) / dt
# ... idem para a largura/altura (dw, dh)

# Projeção: posição = última_posição + velocidade × quadros_faltando
new_x1 = int(last_real['bbox'][0] + dx * frames_missing)
new_y1 = int(last_real['bbox'][1] + dy * frames_missing)
```

Matematicamente, é uma **extrapolação linear de primeira ordem** (modelo de velocidade constante). A caixa interpolada recebe confiança decrescente (`max(0.3, conf − 0.1 × frames_missing)`) e é marcada com `'interpolated': True`, para ser pintada de **laranja** na visualização e identificada com o sufixo `_interp` ao salvar.

### 6.2. Cuidados de implementação

- **Detecções interpoladas não entram no histórico.** Isso é deliberado: garante que `frames_missing` seja sempre medido a partir da última detecção *real*, fazendo o contador crescer de fato e o teto de 5 quadros realmente disparar.
- **Caixas degeneradas são descartadas** (largura/altura < 5 px), evitando "caixas-fantasma" quando a bicicleta já saiu de cena.
- **`_prune_dead_tracks()`** remove rastros que sumiram há mais de 5 quadros, impedindo acúmulo de histórico antigo.

> **Resultado** (de `MELHORIAS_DETECCAO.md`): uma sequência que antes tinha lacunas (`✓ ✓ ✗ ✗ ✗ ✓`) passa a ser contínua (`✓ ✓ interp interp interp ✓`).

---

## 7. Etapa 3 — Recorte e janela temporal deslizante

Detectada a bicicleta, recorta-se apenas a região de interesse (a própria bike), descartando o fundo — o que reduz ruído para o classificador:

```python
def crop_bike(self, frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]   # fatiamento NumPy: O(1), sem cópia desnecessária
```

Os recortes de cada `track_id` são acumulados em um **buffer temporal**. O método `process_video_realtime` (em `main.py`) mantém uma **janela deslizante** dos últimos `window_size` quadros:

```python
frame_buffer.setdefault(track_id, []).append(cropped_bike)

if len(frame_buffer[track_id]) >= window_size:          # janela cheia?
    frame_buffer[track_id] = frame_buffer[track_id][-window_size:]  # mantém só os últimos N
    if frame_count % stride == 0:                       # analisa a cada 'stride' quadros
        prediction = self.trick_classifier.predict_sequence(frame_buffer[track_id], aggregate='average')
```

### 7.1. Janela (`window_size`) e passo (`stride`)

- **`window_size`** (padrão 45): quantos quadros formam o "clipe" analisado. Deve cobrir a duração típica de um giro.
- **`stride`** (padrão 5): de quantos em quantos quadros o classificador é executado. Um *stride* maior acelera o processamento (analisa menos vezes); um menor dá resposta mais granular no tempo.

Esse esquema de **janela deslizante** (*sliding window*) é a técnica clássica para transformar um problema de "detecção no tempo" em uma sequência de classificações de clipes curtos.

### 7.2. Propagação da detecção

Quando um "360" é detectado em um quadro, a marcação é propagada para os quadros seguintes da janela, de forma que a legenda permaneça estável durante todo o giro (em vez de piscar):

```python
trick_detected_frames.add(frame_count)
for i in range(frame_count + 1, frame_count + window_size + 1):
    trick_detected_frames.add(i)
```

---

## 8. Etapa 4 — Classificação da manobra

Esta é a etapa central. O projeto oferece **duas abordagens**, ambas implementadas.

### 8.1. Abordagem A — CNN por quadro + análise de movimento (em produção)

Implementada em `trick_classifier.py`, é a usada por `main.py`. Ela **combina** um classificador de aparência (rede neural) com descritores clássicos de movimento. A intuição: um "360" se distingue por **muita mudança visual ao longo do tempo**, enquanto um empinar ou andar reto mantém a aparência estável.

#### 8.1.1. O classificador de aparência (MobileNetV2 + transfer learning)

```python
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False        # congela o backbone pré-treinado

self.model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./127.5, offset=-1),   # normaliza para [-1, 1]
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='softmax')    # [normal, 360]
])
```

**Conceito — *transfer learning*:** em vez de treinar uma CNN do zero (precisaria de milhões de imagens), reaproveita-se a MobileNetV2 já treinada na ImageNet. Suas camadas convolucionais já sabem extrair bordas, texturas e formas; congelamos (`trainable = False`) essas camadas e treinamos apenas o "cabeçote" (as camadas densas finais) para a nossa tarefa de 2 classes. Isso permite aprender com **centenas** de imagens em vez de milhões.

> Há também `build_cnn_model()`, uma CNN treinada do zero (4 blocos Conv→Pool→BatchNorm), oferecida como alternativa didática.

#### 8.1.2. Os descritores clássicos de movimento

A rede acima vê quadros isolados. A informação temporal vem de duas medidas clássicas, calculadas sobre a sequência inteira:

**(a) Diferença entre quadros** (`compute_visual_change`) — mede o quanto a imagem muda de um quadro para o outro. Para cada par de quadros consecutivos, convertidos para 64×64 em escala de cinza:

```python
diff = np.mean(np.abs(resized[i] - resized[i-1])) / 255.0
```

É a média do valor absoluto da diferença pixel-a-pixel, normalizada para [0,1]. Calculam-se três variantes: movimento de curto prazo (quadros vizinhos), mudança de longo prazo (a cada 1/4 da janela) e diferença início↔fim.

**(b) Variância de histograma** (`compute_histogram_variance`) — mede o quanto a *distribuição de intensidades* muda ao longo do clipe. Compara histogramas de quadros amostrados usando correlação:

```python
hist = cv2.calcHist([img], [0], None, [32], [0, 256])
hist = hist.flatten() / (hist.sum() + 1e-7)         # normaliza
corr = cv2.compareHist(hist_anterior, hist_atual, cv2.HISTCMP_CORREL)
variance_score = 1.0 - max(0.0, avg_corr)           # alta variância → giro
```

Quando a bicicleta gira, partes diferentes ficam visíveis e os histogramas descorrelacionam; quando ela está estável, os histogramas permanecem similares (correlação ≈ 1, variância ≈ 0).

#### 8.1.3. Fusão das evidências

A pontuação de movimento combina as três métricas com pesos:

```python
combined = (motion_score * 0.3) + (change_score * 0.35) + (hist_variance * 0.35)
```

E a decisão final **funde a confiança da CNN com a pontuação de movimento** (`predict_sequence`):

```python
combined_360 = (cnn_360_conf * 0.4) + (motion_score * 0.6)   # movimento pesa mais
if combined_360 > 0.5:
    final_class = '360'
```

Note que o **movimento tem peso maior (0.6)** que a aparência da CNN (0.4) — coerente com a natureza temporal da manobra. Essa fusão híbrida (aprendizado profundo + visão computacional clássica) é o método principal do trabalho.

### 8.2. Abordagem B — Modelos puramente sequenciais (CNN+LSTM / Conv3D)

Implementada em `sequence_trick_classifier.py`, aprende o padrão espaço-temporal diretamente dos dados, sem descritores manuais.

#### CNN + LSTM

```python
sequence_input = layers.Input(shape=(15, 224, 224, 3))   # 15 quadros
x = layers.TimeDistributed(frame_features)(sequence_input)  # MobileNetV2 em cada quadro
x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x) # dependências temporais
x = layers.LSTM(64, dropout=0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(2, activation='softmax')(x)
```

- **`TimeDistributed`** aplica a *mesma* CNN (MobileNetV2) a cada um dos 15 quadros, produzindo um vetor de características por quadro.
- As camadas **LSTM** (*Long Short-Term Memory*) processam essa sequência de vetores, aprendendo a ordem temporal — é o componente que "entende" que frente→lateral→traseira→frente caracteriza um giro.

#### Conv3D (alternativa)

Em vez de separar espaço e tempo, a `Conv3D` aplica convoluções tridimensionais `(tempo, altura, largura)` diretamente sobre o volume de quadros, aprendendo características espaço-temporais conjuntas. É mais pesada, indicada para conjuntos de dados maiores.

### 8.3. Por que duas abordagens?

A abordagem A (híbrida) é robusta com **pouquíssimos dados** porque os descritores de movimento não precisam ser aprendidos. A abordagem B (sequencial pura) tende a ser superior **com muitos dados**, pois aprende o padrão sem suposições manuais. No estágio atual do projeto, com um conjunto pequeno, a abordagem híbrida é a usada em produção.

---

## 9. Etapa 5 — Anotação e geração do vídeo de saída

Por fim, cada quadro é anotado com OpenCV e escrito no vídeo de saída via `cv2.VideoWriter`:

```python
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
...
color = (0, 0, 255) if is_trick else (0, 255, 0)   # vermelho se 360, verde se normal
cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

if is_trick:
    cv2.putText(annotated_frame, "MANOBRA 360 DETECTADA", (50, 50), ...)
    cv2.putText(annotated_frame, f"Confianca: {current_confidence:.2%}", (50, 90), ...)

out.write(annotated_frame)
```

Convenção visual:

| Cor | Significado |
|---|---|
| 🟢 Verde | Bicicleta detectada, sem manobra |
| 🟠 Laranja | Detecção interpolada (no modo de extração) |
| 🔴 Vermelho | Manobra "360" em andamento |

O artefato final é o `resultado.mp4`.

---

## 10. Preparação do dataset

O conjunto de dados foi construído a partir dos próprios vídeos, em duas organizações distintas — uma para cada abordagem de classificação.

### 10.1. Dataset por imagem (abordagem A)

```
dataset/
├── train/
│   ├── normal/   (102 imagens)   # bicicleta em pose comum
│   └── 360/      (202 imagens)   # quadros do giro
└── validation/
    ├── normal/   (19 imagens)
    └── 360/      (77 imagens)
```

### 10.2. Dataset por sequência (abordagem B)

```
sequences_dataset/
├── 360/                          # POSITIVOS (clipes de giro)
│   ├── 360_de_frente_seq001/  (22 frames)
│   ├── 360_de_frente_seq002/  (30 frames)
│   ├── 360_de_frente_seq003/  (25 frames)
│   ├── 360_de_frente_seq004/  (61 frames)
│   └── 360_de_frente_seq005/  (58 frames)
└── normal/                       # NEGATIVOS temporais (ex.: andando de lado)
    └── andando_normal_de_lado_seq001/
```

### 10.3. Como os dados são gerados

Os quadros recortados saem do próprio detector, no **modo `extract`**:

```bash
python main.py --mode extract --video videos/360_de_frente.mp4
```

Isso salva, em `bike_frames/`, um recorte por bicicleta detectada (com sufixo `_interp` nos interpolados). Em seguida, **a separação em `normal/` e `360/` é manual** — o `prepare_dataset.py` apenas *verifica* a estrutura (`--check`) e orienta a organização (`--organize`), pois não há rótulo automático.

### 10.4. Gerador de sequências e balanceamento (`sequence_data_generator.py`)

O `SequenceDataGenerator` (subclasse de `keras.utils.Sequence`) é o responsável por entregar lotes de treino balanceados. Características-chave:

- **Três tipos de amostra:** sequências de 360 (positivos temporais), sequências reais de "normal" (negativos temporais fortes, ex. andar de lado) e imagens normais isoladas (transformadas em "sequência estática" — a mesma imagem repetida 15 vezes, um negativo "limpo" sem movimento).
- **Oversampling (`oversample=6`):** como o conjunto de positivos é pequeno, cada sequência de 360 é repetida 6 vezes; combinada com *augmentation*, cada repetição vira um exemplo distinto. Isso compensa o desbalanceamento.
- **Amostragem temporal uniforme:** clipes longos são reduzidos para `sequence_length=15` quadros escolhidos com `np.linspace`; clipes curtos são preenchidos repetindo o último quadro.
- **Augmentation coerente no tempo:** rotação, *flip* e brilho são sorteados **uma vez por sequência** e aplicados a *todos* os quadros do clipe, preservando a coerência temporal (não faria sentido espelhar só metade do giro).

---

## 11. Treinamento dos modelos

### 11.1. Classificador por imagem (abordagem A)

`train_model.py` usa o `ImageDataGenerator` do Keras para *augmentation* on-the-fly (rotação 20°, deslocamentos, *flip* horizontal, zoom) e treina com:

- **Função de perda:** entropia cruzada categórica (`categorical_crossentropy`) — padrão para classificação multiclasse com saída *softmax*.
- **Otimizador:** Adam, `learning_rate=0.001`.
- **Callbacks:** `EarlyStopping` (paciência 10, restaura melhores pesos), `ReduceLROnPlateau` (reduz a taxa de aprendizado em platôs) e `ModelCheckpoint` (salva o melhor modelo).

```bash
python train_model.py --train_dir dataset/train --val_dir dataset/validation --epochs 50
```

Saída: `trick_classifier_model.h5` (+ `best_trick_model.h5`).

### 11.2. Classificador sequencial (abordagem B)

`train_sequence_model.py` instancia o `SequenceTrickClassifier` e treina via `SequenceDataGenerator`:

- **Otimizador:** Adam, `learning_rate=0.0005` (menor, pois há LSTM).
- **Batch pequeno (8):** sequências de 15×224×224×3 consomem muita memória.
- **Callbacks** análogos, com paciência maior (15) por ser um problema mais difícil.

```bash
python train_sequence_model.py \
    --sequences_360_train sequences_dataset/360 \
    --normal_train dataset/train/normal \
    --normal_sequences_train sequences_dataset/normal \
    --epochs 50 --batch_size 8 --sequence_length 15 --model_type cnn_lstm
```

Saída: `sequence_trick_classifier_cnn_lstm.h5` + `_config.json` + `_classes.txt`.

---

## 12. Tutorial: como reproduzir o resultado do zero

Esta seção é um passo a passo replicável para construir um sistema equivalente.

### Passo 0 — Ambiente

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt    # opencv, numpy, tensorflow, ultralytics, pillow, matplotlib
```

O modelo YOLO (`yolov8m.pt`) é baixado automaticamente no primeiro uso.

### Passo 1 — Detectar e extrair os recortes da bicicleta

```bash
python main.py --mode extract --video videos/SEU_VIDEO.mp4
```

Gera `bike_frames/` (recortes) e um vídeo com as caixas desenhadas. **Princípio aprendido:** use um detector pré-treinado (YOLO) filtrando a classe de interesse, e rastreie (ByteTrack) para manter a identidade no tempo.

### Passo 2 — Montar o conjunto de dados

1. Crie a estrutura: `python train_model.py --setup`
2. Separe manualmente os recortes de `bike_frames/` em `dataset/train/{normal,360}` e `dataset/validation/{normal,360}`.
3. Para a abordagem sequencial, agrupe os quadros de cada giro em subpastas dentro de `sequences_dataset/360/`.
4. Verifique: `python prepare_dataset.py --check`.

**Princípio aprendido:** o rótulo de "ação" exige contexto temporal; por isso o dataset sequencial agrupa *clipes*, não imagens soltas. Mantenha negativos temporais reais (movimentos que **não** são 360) para o modelo não confundir "movimento" com "giro".

### Passo 3 — Treinar o classificador

```bash
# Abordagem A (imagem + movimento)
python train_model.py --train_dir dataset/train --val_dir dataset/validation --epochs 50

# Abordagem B (sequência CNN+LSTM)
python train_sequence_model.py --epochs 50 --model_type cnn_lstm
```

**Princípio aprendido:** com poucos dados, prefira *transfer learning* (congele o backbone) e use *augmentation* + *callbacks* para evitar *overfitting*.

### Passo 4 — Detectar manobras no vídeo completo

```bash
python main.py --mode detect --video videos/SEU_VIDEO.mp4 --output resultado.mp4 \
    --window_size 45 --stride 5
```

**Princípio aprendido:** percorra o vídeo com janela deslizante, classifique cada clipe e funda evidências de aparência (CNN) e de movimento (diferença de quadros + histograma) para decidir.

### Passo 5 — Análise resumida (opcional)

```bash
python main.py --mode analyze --video videos/SEU_VIDEO.mp4
```

Imprime a porcentagem de quadros com manobra detectada, sem gerar vídeo.

---

## 13. Parâmetros e ajustes finos

| Parâmetro | Onde | Padrão | Efeito de aumentar |
|---|---|---|---|
| `confidence_threshold` | `BikeDetector` | 0.4 | Menos detecções falsas, mais lacunas durante o giro |
| `imgsz` | `detect_bikes` | 1280 | Mais sensibilidade, mais custo computacional |
| `max_frames_missing` | `BikeDetector` | 5 | Interpola lacunas maiores, risco de caixas imprecisas |
| `window_size` | `process_video_realtime` | 45 | Mais contexto temporal, resposta mais lenta |
| `stride` | `process_video_realtime` | 5 | Processamento mais rápido, menos granularidade temporal |
| Limiar de decisão `> 0.5` | `predict_sequence` | 0.5 | Mais conservador (menos falsos positivos) |
| Pesos da fusão `0.4 / 0.6` | `predict_sequence` | CNN/movimento | Privilegia aparência vs. movimento |
| `sequence_length` | sequencial | 15 | Mais quadros por clipe, mais memória |
| `oversample` | gerador | 6 | Mais repetição dos positivos (compensa dataset pequeno) |

**Receitas rápidas de *troubleshooting*** (de `README.md` e `MELHORIAS_DETECCAO.md`):

- *YOLO não detecta a bicicleta* → reduza `confidence_threshold`.
- *Sequências com lacunas* → reduza `conf` para 0.3, aumente `max_frames_missing`, ou use um YOLO maior (`yolov8x.pt`).
- *Processamento lento* → aumente `stride` ou reduza `window_size`.
- *Erro de memória no treino* → reduza `batch_size`.

---

## 14. Limitações e trabalhos futuros

- **Rótulo manual:** a separação em `normal/360` é feita à mão; não há rotulagem automática.
- **Conjunto pequeno e específico:** os giros são majoritariamente "360 de frente"; generalização para outros ângulos/manobras exige mais dados.
- **Interpolação linear:** assume velocidade constante; movimentos erráticos podem gerar caixas imprecisas além de ~5 quadros.
- **Dependência de limiares manuais** na fusão híbrida (pesos 0.4/0.6, limiares de movimento), ajustados empiricamente.
- **Caminhos futuros:** ampliar o dataset, adotar plenamente o modelo CNN+LSTM, estabilização de vídeo no pré-processamento, e estimar `[t_início, t_fim]` da manobra com mais precisão (suavização temporal das predições).

---

## 15. Conclusão

O trabalho demonstra um pipeline completo de **detecção de ação em vídeo** construído com ferramentas acessíveis. A solução combina, de forma pragmática, o melhor de dois mundos: a **detecção robusta por aprendizado profundo** (YOLOv8 + ByteTrack) para localizar e rastrear a bicicleta, e uma **classificação temporal híbrida** que une uma CNN por *transfer learning* (MobileNetV2) a descritores clássicos de movimento (diferença de quadros e variância de histograma). A engenharia em torno do problema — limiar de confiança calibrado, interpolação de lacunas, janela deslizante e fusão ponderada de evidências — é o que torna a detecção do "360" estável mesmo com um conjunto de dados modesto. O relatório também documenta a via alternativa CNN+LSTM/Conv3D, que aponta o caminho natural de evolução quando houver mais dados disponíveis.

---

## 16. Referências

- **YOLOv8 — Ultralytics.** Documentação oficial: <https://docs.ultralytics.com/>
- **ByteTrack:** Zhang, Y. et al. *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV, 2022.
- **MobileNetV2:** Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR, 2018.
- **LSTM:** Hochreiter, S.; Schmidhuber, J. *Long Short-Term Memory.* Neural Computation, 1997.
- **COCO Dataset:** Lin, T.-Y. et al. *Microsoft COCO: Common Objects in Context.* ECCV, 2014.
- **OpenCV:** <https://docs.opencv.org/>
- **TensorFlow/Keras:** <https://www.tensorflow.org/> · <https://keras.io/>
- Documentação interna do projeto: `README.md`, `README_SEQUENCE_TRAINING.md`, `MELHORIAS_DETECCAO.md`, `GUIA_RAPIDO.md`.
