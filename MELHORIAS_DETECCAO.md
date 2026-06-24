# Melhorias na Detecção de Frames

## Problema Identificado

Durante a extração de frames, havia **gaps** (lacunas) no meio do movimento do 360, onde alguns frames não eram detectados, resultando em sequências incompletas.

## Soluções Implementadas

### 1. **Redução do Threshold de Confiança**
- **Antes**: `confidence_threshold=0.5`
- **Agora**: `confidence_threshold=0.4`
- **Motivo**: Durante movimentos rápidos como o 360, a bike pode aparecer desfocada ou em ângulos difíceis, reduzindo a confiança da detecção. Um threshold menor captura mais frames.

### 2. **Parâmetros Otimizados do YOLO**
```python
results = self.model.track(
    frame, 
    persist=True, 
    verbose=False, 
    imgsz=1280,           # Resolução maior para melhor detecção
    conf=0.4,             # Threshold reduzido
    iou=0.5,              # IoU para matching de boxes
    tracker='bytetrack.yaml'  # Tracker ByteTrack (mais robusto)
)
```

### 3. **Interpolação de Frames Perdidos**
Implementada função `_interpolate_missing_tracks()` que:
- Mantém histórico de detecções por track_id
- Quando uma bike não é detectada por até 5 frames consecutivos
- Interpola a posição baseada na velocidade e direção anteriores
- Adiciona detecções interpoladas com confiança reduzida

**Como funciona:**
```python
# Calcula velocidade do movimento
dx = (última_posição - posição_anterior) / frames_entre_elas
dy = ...

# Projeta nova posição
nova_posição = última_posição + (velocidade * frames_perdidos)
```

### 4. **Histórico de Tracking**
- Cada bike detectada mantém histórico de suas posições
- Permite análise temporal do movimento
- Facilita interpolação e suavização

### 5. **Visualização de Frames Interpolados**
- Frames detectados: **Verde**
- Frames interpolados: **Laranja** com label "(interp)"
- Arquivos salvos com sufixo `_interp` para identificação

## Resultados Esperados

### Antes
```
Frame 100: ✓ Detectado
Frame 101: ✓ Detectado
Frame 102: ✗ GAP
Frame 103: ✗ GAP
Frame 104: ✗ GAP
Frame 105: ✓ Detectado
```

### Depois
```
Frame 100: ✓ Detectado
Frame 101: ✓ Detectado
Frame 102: ✓ Interpolado
Frame 103: ✓ Interpolado
Frame 104: ✓ Interpolado
Frame 105: ✓ Detectado
```

## Como Usar

### Extrair Frames com Melhorias
```bash
python main.py --mode extract --video videos/360_frente_pf.mp4
```

### Parâmetros Ajustáveis

No arquivo `bike_detector.py`:
```python
self.max_frames_missing = 5  # Máximo de frames para interpolar
```

Você pode ajustar:
- `max_frames_missing`: Quantos frames consecutivos podem ser interpolados (padrão: 5)
- `confidence_threshold`: Threshold mínimo de confiança (padrão: 0.4)

## Benefícios

1. **Sequências Mais Completas**: Menos gaps durante movimentos rápidos
2. **Melhor Treinamento**: Mais frames = melhor aprendizado do modelo
3. **Tracking Robusto**: ByteTrack mantém IDs consistentes
4. **Transparência**: Frames interpolados são claramente marcados
5. **Flexível**: Parâmetros ajustáveis conforme necessidade

## Limitações

- Interpolação funciona melhor para movimentos suaves
- Movimentos muito erráticos podem ter interpolação imprecisa
- Após 5 frames sem detecção, o tracking é perdido
- Frames interpolados têm confiança reduzida (não ideal para treino se houver muitos)

## Próximos Passos

Se ainda houver gaps:
1. Reduzir ainda mais o `confidence_threshold` (ex: 0.3)
2. Aumentar `max_frames_missing` (ex: 7-10 frames)
3. Usar modelo YOLO maior (yolov8x.pt) para melhor detecção
4. Aplicar pré-processamento no vídeo (estabilização, aumento de contraste)
