# Treinamento com Sequências de Frames

## Visão Geral

A nova arquitetura de treinamento foi modificada para trabalhar com **sequências de frames** para detectar manobras 360, enquanto usa imagens individuais para a classe "normal".

## Estrutura de Dados

### Para 360s (Sequências)
```
sequences_dataset/
└── 360/
    ├── 360_de_frente_seq001/
    │   ├── bike_frame_000147_det_0.jpg
    │   ├── bike_frame_000148_det_0.jpg
    │   └── ...
    ├── 360_de_frente_seq002/
    └── ...
```

### Para Normal (Imagens Individuais)
```
dataset/
├── train/
│   └── normal/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── validation/
    └── normal/
        ├── image1.jpg
        └── ...
```

## Arquivos Criados

1. **`sequence_data_generator.py`**: Gerador de dados customizado que:
   - Carrega sequências de frames de subpastas para 360
   - Carrega imagens individuais para normal (e as replica em sequência)
   - Aplica data augmentation
   - Balanceia as classes automaticamente

2. **`sequence_trick_classifier.py`**: Classificador com duas arquiteturas:
   - **CNN + LSTM**: Usa MobileNetV2 para extrair features de cada frame, depois LSTM para capturar dependências temporais
   - **Conv3D**: Processa a sequência diretamente com convoluções 3D

3. **`train_sequence_model.py`**: Script de treinamento principal

## Como Usar

### Treinamento Básico

```bash
python train_sequence_model.py
```

Isso usará os diretórios padrão:
- Sequências 360: `sequences_dataset/360`
- Imagens normais: `dataset/train/normal`

### Treinamento com Parâmetros Customizados

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

### Parâmetros Disponíveis

- `--sequences_360_train`: Diretório com sequências de 360 para treino
- `--normal_train`: Diretório com imagens normais para treino
- `--sequences_360_val`: Diretório com sequências de 360 para validação (opcional)
- `--normal_val`: Diretório com imagens normais para validação
- `--epochs`: Número de épocas (padrão: 50)
- `--batch_size`: Tamanho do batch (padrão: 8)
- `--sequence_length`: Número de frames por sequência (padrão: 15)
- `--model_type`: Tipo de modelo - `cnn_lstm` ou `conv3d` (padrão: cnn_lstm)

## Tipos de Modelo

### CNN + LSTM (Recomendado)
- Usa transfer learning com MobileNetV2
- Mais eficiente e rápido para treinar
- Melhor para datasets menores
- Captura bem dependências temporais

### Conv3D
- Processa sequências diretamente
- Mais pesado computacionalmente
- Pode ser melhor com datasets grandes
- Aprende features espaço-temporais diretamente

## Predição

```python
from sequence_trick_classifier import SequenceTrickClassifier

# Carregar modelo
classifier = SequenceTrickClassifier()
classifier.load_model('sequence_trick_classifier_cnn_lstm.h5')

# Predizer sequência
image_sequence = [frame1, frame2, frame3, ...]  # Lista de imagens
result = classifier.predict_sequence(image_sequence)

print(f"Classe: {result['class']}")
print(f"Confiança: {result['confidence']:.2%}")
print(f"Todas as predições: {result['all_predictions']}")
```

## Vantagens da Nova Arquitetura

1. **Contexto Temporal**: Captura o movimento completo do 360, não apenas frames isolados
2. **Mais Robusto**: Menos sensível a frames individuais ambíguos
3. **Balanceamento Automático**: O gerador balanceia automaticamente as classes
4. **Data Augmentation**: Aplica augmentation em sequências para melhor generalização
5. **Flexível**: Suporta sequências de tamanhos variados (ajusta automaticamente)

## Callbacks e Otimizações

O treinamento inclui:
- **EarlyStopping**: Para quando não há melhoria (patience=15)
- **ReduceLROnPlateau**: Reduz learning rate quando estagnado (patience=7)
- **ModelCheckpoint**: Salva o melhor modelo baseado em accuracy

## Outputs do Treinamento

- `sequence_trick_classifier_cnn_lstm.h5`: Modelo final
- `best_sequence_trick_model.h5`: Melhor modelo durante treinamento
- `sequence_trick_classifier_cnn_lstm_config.json`: Configuração do modelo
- `sequence_trick_classifier_cnn_lstm_classes.txt`: Nomes das classes
