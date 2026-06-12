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

```bash
python train_model.py --train_dir dataset/train --val_dir dataset/validation --epochs 50
```

### 7️⃣ Detectar Manobras

```bash
python main.py --mode detect --video videos/360_de_lado.mp4 --output resultado.mp4
```

## 🧪 Testes do Modelo

Durante o desenvolvimento, foram realizados diversos testes para validar o classificador de manobras:

- **Separação de manobras**: Foi necessário separar manualmente os frames de bicicletas em posição normal e bicicletas executando o 360, utilizando vídeos de diferentes ângulos e fontes para garantir variedade no dataset.
- **Ajuste de confiança**: O threshold de confiança foi ajustado de 0.7 para 0.6 após análise dos logs de predição, que mostravam que o modelo classificava corretamente os 360 mas com confiança entre 0.60 e 0.63.
- **Ajuste no dataset**: O dataset inicial continha apenas 36 imagens, resultando em baixa acurácia. Após aumentar para 136 imagens (64 normal + 72 de 360), combinando frames de múltiplos vídeos, o modelo passou a generalizar melhor e detectar as manobras de forma consistente.

## 🔧 Comandos Úteis

```bash
# Verificar estrutura do dataset
python prepare_dataset.py --check

# Testar modelo treinado
python prepare_dataset.py --test

# Ver guia completo
python prepare_dataset.py --guidex
```
