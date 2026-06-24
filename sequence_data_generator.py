import numpy as np
import cv2
import os
from tensorflow import keras


class SequenceDataGenerator(keras.utils.Sequence):
    """Gerador de dados que carrega sequências de frames para 360 e imagens individuais para normal."""
    
    def __init__(self, sequences_360_dir, normal_images_dir=None, normal_sequences_dir=None,
                 batch_size=8, img_height=224, img_width=224, sequence_length=15,
                 shuffle=True, augment=False, oversample=6):
        """
        Args:
            sequences_360_dir: Diretório com subpastas de sequências de 360
            normal_images_dir: Diretório com imagens de bikes normais (replicadas em sequência estática)
            normal_sequences_dir: Diretório com subpastas de sequências REAIS de movimento normal
                                  (ex.: andar de lado) — negativos temporais fortes
            batch_size: Tamanho do batch
            img_height: Altura das imagens
            img_width: Largura das imagens
            sequence_length: Número de frames por sequência
            shuffle: Se deve embaralhar os dados
            augment: Se deve aplicar data augmentation
            oversample: Fator de repetição das sequências por classe (compensa dataset pequeno)
        """
        self.sequences_360_dir = sequences_360_dir
        self.normal_images_dir = normal_images_dir
        self.normal_sequences_dir = normal_sequences_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        self.oversample = max(1, oversample)

        # Carregar lista de sequências 360 (positivos)
        self.sequences_360 = self._list_sequences(sequences_360_dir, min_frames=sequence_length)

        # Carregar lista de sequências REAIS de normal (negativos temporais)
        self.normal_sequences = self._list_sequences(normal_sequences_dir, min_frames=1)

        # Carregar lista de imagens normais individuais (negativos estáticos)
        self.normal_images = []
        if normal_images_dir and os.path.exists(normal_images_dir):
            self.normal_images = [os.path.join(normal_images_dir, f)
                                 for f in os.listdir(normal_images_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Criar lista de amostras balanceada entre 360 e normal
        self.samples = self._build_samples()

        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)

        n_pos = sum(1 for label, _ in self.samples if label == '360')
        print(f"Dataset carregado:")
        print(f"  - Sequências 360 (únicas): {len(self.sequences_360)}")
        print(f"  - Sequências normais reais (únicas): {len(self.normal_sequences)}")
        print(f"  - Imagens normais individuais: {len(self.normal_images)}")
        print(f"  - Amostras totais: {len(self.samples)} ({n_pos} 360 / {len(self.samples) - n_pos} normal)")

    def _list_sequences(self, root_dir, min_frames):
        """Lista subpastas que contêm pelo menos min_frames imagens."""
        sequences = []
        if root_dir and os.path.exists(root_dir):
            for seq_folder in sorted(os.listdir(root_dir)):
                seq_path = os.path.join(root_dir, seq_folder)
                if os.path.isdir(seq_path):
                    frames = [f for f in os.listdir(seq_path)
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if len(frames) >= min_frames:
                        sequences.append(seq_path)
        return sequences

    def _build_samples(self):
        """Monta uma lista de amostras balanceada entre as classes.

        Positivos (360) são oversampled (com augmentation viram exemplos distintos).
        Negativos são divididos entre sequências reais de normal e imagens estáticas,
        garantindo que os negativos temporais reais fiquem bem representados.
        """
        samples = []
        n_360 = len(self.sequences_360)
        if n_360 == 0:
            return samples

        samples_per_class = n_360 * self.oversample

        # Positivos
        for i in range(samples_per_class):
            samples.append(('360', self.sequences_360[i % n_360]))

        # Negativos
        has_seq = len(self.normal_sequences) > 0
        has_img = len(self.normal_images) > 0

        if has_seq and has_img:
            half = samples_per_class // 2
            for i in range(half):
                samples.append(('normal_seq', self.normal_sequences[i % len(self.normal_sequences)]))
            for _ in range(samples_per_class - half):
                idx = np.random.randint(len(self.normal_images))
                samples.append(('normal_img', self.normal_images[idx]))
        elif has_seq:
            for i in range(samples_per_class):
                samples.append(('normal_seq', self.normal_sequences[i % len(self.normal_sequences)]))
        elif has_img:
            for _ in range(samples_per_class):
                idx = np.random.randint(len(self.normal_images))
                samples.append(('normal_img', self.normal_images[idx]))

        return samples
    
    def __len__(self):
        """Retorna o número de batches por época."""
        return int(np.ceil(len(self.samples) / self.batch_size))
    
    def __getitem__(self, index):
        """Gera um batch de dados."""
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        X = []
        y = []
        
        for idx in batch_indexes:
            label, path = self.samples[idx]

            if label in ('360', 'normal_seq'):
                # Carregar sequência real de frames
                sequence = self._load_sequence(path)
            else:
                # Criar "sequência" estática a partir de uma imagem normal
                sequence = self._load_normal_as_sequence(path)

            X.append(sequence)
            y.append(1 if label == '360' else 0)
        
        X = np.array(X, dtype=np.float32)
        y = keras.utils.to_categorical(y, num_classes=2)
        
        return X, y
    
    def _load_sequence(self, seq_path):
        """Carrega uma sequência de frames de um diretório."""
        frames = sorted([f for f in os.listdir(seq_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        # Selecionar frames uniformemente distribuídos
        if len(frames) > self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length, dtype=int)
            selected_frames = [frames[i] for i in indices]
        else:
            selected_frames = frames[:self.sequence_length]
            # Preencher com o último frame se necessário
            while len(selected_frames) < self.sequence_length:
                selected_frames.append(frames[-1])
        
        # Parâmetros de augmentation sorteados UMA vez por sequência, para preservar
        # a coerência temporal (todo o clipe sofre a mesma rotação/flip/brilho).
        params = self._get_augment_params() if self.augment else None

        sequence = []
        for frame_name in selected_frames:
            frame_path = os.path.join(seq_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height))

            if params is not None:
                img = self._apply_augment(img, params)

            # MobileNetV2 espera entrada em [0,255]; a normalização para [-1,1]
            # é feita pela camada Rescaling(1/127.5, -1) DENTRO do modelo.
            # NÃO dividir por 255 aqui (dupla normalização colapsaria a entrada
            # para ~-1 e o modelo não aprenderia nada).
            img = img.astype(np.float32)
            sequence.append(img)

        return np.array(sequence)

    def _load_normal_as_sequence(self, img_path):
        """Carrega uma imagem normal e cria uma sequência estática (sem movimento)."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))

        params = self._get_augment_params() if self.augment else None
        if params is not None:
            img = self._apply_augment(img, params)

        # Manter [0,255] (a normalização é feita pela camada Rescaling do modelo)
        img = img.astype(np.float32)
        # Mesma imagem repetida -> sequência sem movimento (negativo limpo)
        return np.array([img.copy() for _ in range(self.sequence_length)])

    def _get_augment_params(self):
        """Sorteia parâmetros de augmentation para uma sequência inteira."""
        return {
            'rotate': np.random.random() > 0.5,
            'angle': np.random.uniform(-15, 15),
            'flip': np.random.random() > 0.5,
            'bright': np.random.random() > 0.5,
            'factor': np.random.uniform(0.8, 1.2),
        }

    def _apply_augment(self, img, params):
        """Aplica os mesmos parâmetros de augmentation a um frame (uint8 RGB)."""
        if params['rotate']:
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), params['angle'], 1.0)
            img = cv2.warpAffine(img, M, (w, h))

        if params['flip']:
            img = cv2.flip(img, 1)

        if params['bright']:
            img = np.clip(img.astype(np.float32) * params['factor'], 0, 255).astype(np.uint8)

        return img
    
    def on_epoch_end(self):
        """Atualiza índices após cada época."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
