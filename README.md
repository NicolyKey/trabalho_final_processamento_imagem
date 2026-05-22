## Trabalho Final de Processamento de Imagens

### Objetivo

O trabalho consiste na análise de vídeos de manobras de bicicletas e a identificação de uma manobra específica (360 graus).

### Descrição do Sistema

Este projeto utiliza duas tecnologias principais:

1. **YOLO (YOLOv8)** - Para detecção de bicicletas nos vídeos
   - Detecta e isola bicicletas ignorando o fundo
   - Extrai coordenadas (bounding boxes) das bicicletas
   - Confiança mínima configurável

2. **Keras/TensorFlow** - Para classificação de manobras
   - Classifica se a bicicleta está fazendo um 360 graus
   - Usa transfer learning com MobileNetV2 ou CNN customizada
   - Análise temporal com janelas deslizantes

### Estrutura do Projeto

```
trabalho_final_pi/
├── videos/                          # Vídeos de entrada
│   └── crianca_bicicleta.mp4
├── bike_detector.py                 # Detector de bicicletas com YOLO
├── trick_classifier.py              # Classificador de manobras com Keras
├── train_model.py                   # Script para treinar o classificador
├── main.py                          # Script principal de execução
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Este arquivo
```

### Instalação

1. Clone o repositório e navegue até a pasta do projeto

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. O modelo YOLO será baixado automaticamente na primeira execução

### Uso

#### 1. Extrair Frames de Bicicletas

Para extrair apenas os frames onde bicicletas foram detectadas:

```bash
python main.py --mode extract --video videos/crianca_bicicleta.mp4
```

Isso irá:
- Criar pasta `bike_frames/` com imagens recortadas das bicicletas
- Criar vídeo `output_detected.mp4` com bounding boxes

#### 2. Treinar o Classificador de Manobras

**IMPORTANTE**: Antes de treinar, você precisa ter imagens organizadas!

**Passo 2.1**: Crie a estrutura de diretórios:

```bash
python train_model.py --setup
```

**Passo 2.2**: Organize suas imagens manualmente:

Você precisa separar as imagens extraídas da pasta `bike_frames/` em:
- `dataset/train/normal/` - Imagens de bicicletas em posição normal
- `dataset/train/360/` - Imagens de bicicletas fazendo 360
- `dataset/validation/normal/` - Imagens de validação normais (20% do total)
- `dataset/validation/360/` - Imagens de validação com 360 (20% do total)

**Passo 2.3**: Verifique se o dataset está correto:

```bash
python prepare_dataset.py --check
```

**Passo 2.4**: Treine o modelo:

```bash
python train_model.py --train_dir dataset/train --val_dir dataset/validation --epochs 50
```

Opções de treinamento:
- `--epochs`: Número de épocas (padrão: 50)
- `--batch_size`: Tamanho do batch (padrão: 32)
- `--no_transfer_learning`: Treinar CNN do zero sem transfer learning

#### 3. Detectar Manobras em Vídeos

Após treinar o modelo, execute a detecção completa:

```bash
python main.py --mode detect --video videos/crianca_bicicleta.mp4 --output resultado.mp4
```

Opções:
- `--window_size`: Tamanho da janela temporal (padrão: 30 frames)
- `--stride`: Intervalo entre análises (padrão: 10 frames)
- `--yolo_model`: Modelo YOLO a usar (padrão: yolov8n.pt)
- `--classifier_model`: Modelo de classificação (padrão: trick_classifier_model.h5)

#### 4. Análise Resumida

Para obter estatísticas sem gerar vídeo:

```bash
python main.py --mode analyze --video videos/crianca_bicicleta.mp4
```

### Componentes Principais

#### BikeDetector (`bike_detector.py`)

- **Função**: Detecta bicicletas usando YOLOv8
- **Métodos principais**:
  - `detect_bikes(frame)`: Retorna detecções de bicicletas em um frame
  - `crop_bike(frame, bbox)`: Recorta bicicleta do frame
  - `process_video()`: Processa vídeo completo e extrai frames

#### TrickClassifier (`trick_classifier.py`)

- **Função**: Classifica manobras de bicicleta
- **Arquiteturas disponíveis**:
  - Transfer Learning com MobileNetV2 (recomendado)
  - CNN customizada
- **Métodos principais**:
  - `train()`: Treina o modelo
  - `predict()`: Classifica uma imagem
  - `predict_sequence()`: Classifica sequência de frames

#### BikeManeuverDetector (`main.py`)

- **Função**: Integra detecção e classificação
- **Recursos**:
  - Análise temporal com janelas deslizantes
  - Anotação de vídeos com detecções
  - Estatísticas de detecção

### Parâmetros Configuráveis

- **Confiança YOLO**: Ajuste em `BikeDetector(confidence_threshold=0.5)`
- **Tamanho de imagem**: Ajuste em `TrickClassifier(img_height=224, img_width=224)`
- **Limiar de confiança 360**: Ajuste `prediction['confidence'] > 0.7` em `main.py`

### Outputs Gerados

1. **bike_frames/**: Imagens recortadas das bicicletas detectadas
2. **output_detected.mp4**: Vídeo com bounding boxes das bicicletas
3. **output_with_tricks.mp4**: Vídeo com detecção de manobras 360
4. **trick_classifier_model.h5**: Modelo treinado de classificação
5. **best_trick_model.h5**: Melhor modelo durante treinamento

### Requisitos do Sistema

- Python 3.8+
- GPU recomendada para treinamento (mas funciona em CPU)
- Pelo menos 4GB de RAM
- Espaço em disco para vídeos e modelos

### Troubleshooting

**Problema**: Erro ao treinar - "Nenhuma imagem encontrada" ou erro no `model.fit()`
- **Causa**: Dataset vazio ou mal estruturado
- **Solução**: 
  1. Execute `python prepare_dataset.py --check` para verificar
  2. Certifique-se de ter imagens nas pastas `dataset/train/normal/` e `dataset/train/360/`
  3. As imagens devem estar em formato `.jpg`, `.jpeg` ou `.png`
  4. Você precisa de pelo menos 10-20 imagens por classe

**Problema**: YOLO não detecta bicicletas
- Solução: Reduza `confidence_threshold` em `BikeDetector`

**Problema**: Classificador não detecta 360
- Solução: Treine com mais imagens ou ajuste o limiar de confiança

**Problema**: Processamento muito lento
- Solução: Aumente `stride` ou reduza `window_size`

**Problema**: Erro de memória durante treinamento
- Solução: Reduza `batch_size` para 16 ou 8

### Tecnologias Utilizadas

- **YOLOv8** (Ultralytics): Detecção de objetos em tempo real
- **TensorFlow/Keras**: Framework de deep learning
- **OpenCV**: Processamento de vídeo e imagem
- **NumPy**: Computação numérica

### Autores

Projeto desenvolvido para a disciplina de Processamento de Imagens.