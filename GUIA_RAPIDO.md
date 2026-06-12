# 🚴 Guia Rápido - Detecção de Manobras de Bicicleta

## ⚠️ ERRO COMUM: "Erro ao treinar o modelo"

Se você recebeu um erro ao executar `python train_model.py`, é porque **o dataset está vazio**!

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

## 🎯 Resumo do Problema

O erro acontece porque:
- ❌ As pastas `dataset/train/normal/` e `dataset/train/360/` estão **vazias**
- ❌ O modelo não pode treinar sem imagens

A solução é:
- ✅ Extrair frames com `--mode extract`
- ✅ Organizar manualmente as imagens nas pastas corretas
- ✅ Verificar com `prepare_dataset.py --check`
- ✅ Então treinar o modelo

## 📊 Requisitos Mínimos

Para treinar o modelo você precisa de:
- Pelo menos **10-20 imagens** de bicicletas normais
- Pelo menos **10-20 imagens** de bicicletas fazendo 360
- Imagens em formato `.jpg`, `.jpeg` ou `.png`

## 🔧 Comandos Úteis

```bash
# Verificar estrutura do dataset
python prepare_dataset.py --check

# Testar modelo treinado
python prepare_dataset.py --test

# Ver guia completo
python prepare_dataset.py --guidex
```
