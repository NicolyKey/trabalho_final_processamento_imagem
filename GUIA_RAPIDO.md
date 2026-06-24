# 🚴 Guia Rápido - Detecção de Manobras de Bicicleta

## ✅ Solução Passo a Passo

### 1️⃣ Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Extrair Frames do Vídeo

```bash
python main.py --mode extract --video videos_treinamento/360_de_Lado_treinamento.mp4
```

Isso vai criar a pasta `bike_frames/` com imagens das bicicletas detectadas.

### 3️⃣ Criar Estrutura do Dataset

```bash
python train_model.py --setup
```

Isso cria as pastas:

- `dataset/train/normal/`
- `dataset/train/360/`
- `dataset/validation/normal/`
- `dataset/validation/360/`

### 4️⃣ **IMPORTANTE**: Organizar Imagens Manualmente

Agora você precisa **copiar manualmente** as imagens de `bike_frames/` para as pastas corretas:

1. Abra a pasta `bike_frames/`
2. Veja cada imagem
3. Copie para a pasta apropriada:
   - Se a bicicleta está **normal**: copie para `dataset/train/normal/`
   - Se a bicicleta está fazendo **360**: copie para `dataset/train/360/`

**Dica**: Separe 80% das imagens para `train/` e 20% para `validation/`

### 5️⃣ Verificar Dataset

```bash
python prepare_dataset.py --check
```

Você deve ver algo como:

```
✓ dataset/train/normal
  → 50 imagens encontradas
✓ dataset/train/360
  → 30 imagens encontradas
...
```

### 6️⃣ Treinar o Modelo

#### Opção A: Modelo de Imagens Individuais (Original)

```bash
python train_model.py --train_dir dataset/train --val_dir dataset/validation --epochs 100
```

#### Opção B: Modelo de Sequências (Recomendado para 360)

Para treinar com sequências de frames (melhor para detectar movimento):

```bash
python train_sequence_model.py
```

**Parâmetros disponíveis:**

```bash
python train_sequence_model.py \
  --sequences_360_train sequences_dataset/360 \
  --normal_train dataset/train/normal \
  --normal_val dataset/validation/normal \
  --epochs 50 \
  --batch_size 8 \
  --sequence_length 15 \
  --model_type cnn_lstm
```

**Opções:**
- `--sequences_360_train`: Pasta com subpastas de sequências de 360 (padrão: `sequences_dataset/360`)
- `--normal_train`: Pasta com imagens normais para treino
- `--sequences_360_val`: Sequências de 360 para validação (opcional)
- `--normal_val`: Imagens normais para validação
- `--epochs`: Número de épocas (padrão: 50)
- `--batch_size`: Tamanho do batch (padrão: 8)
- `--sequence_length`: Número de frames por sequência (padrão: 15)
- `--model_type`: `cnn_lstm` (CNN+LSTM) ou `conv3d` (Convolução 3D)

**Estrutura esperada do sequences_dataset:**
```
sequences_dataset/
└── 360/
    ├── sequence_001/
    │   ├── frame_0001.jpg
    │   ├── frame_0002.jpg
    │   └── ...
    ├── sequence_002/
    │   └── ...
    └── ...
```

**Modelos salvos:**
- `sequence_trick_classifier_cnn_lstm.h5` ou `sequence_trick_classifier_conv3d.h5`
- `best_sequence_trick_model.h5` (melhor modelo durante treino)

### 7️⃣ Detectar Manobras

```bash
python main.py --mode detect --video videos/360_de_lado.mp4 --output resultado.mp4
```
## 🔧 Comandos Úteis

```bash
# Verificar estrutura do dataset
python prepare_dataset.py --check

# Testar modelo treinado
python prepare_dataset.py --test

# Ver guia completo
python prepare_dataset.py --guidex
```
