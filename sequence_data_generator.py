import numpy as np
import cv2
import os
from tensorflow import keras


class SequenceDataGenerator(keras.utils.Sequence):
    """Gerador de dados que carrega sequências de frames para 360 e imagens individuais para normal."""
    
    def __init__(self, sequences_360_dir, normal_images_dir, batch_size=8, 
                 img_height=224, img_width=224, sequence_length=15, 
                 shuffle=True, augment=False):
        """
        Args:
            sequences_360_dir: Diretório com subpastas de sequências de 360
            normal_images_dir: Diretório com imagens de bikes normais
            batch_size: Tamanho do batch
            img_height: Altura das imagens
            img_width: Largura das imagens
            sequence_length: Número de frames por sequência
            shuffle: Se deve embaralhar os dados
            augment: Se deve aplicar data augmentation
        """
        self.sequences_360_dir = sequences_360_dir
        self.normal_images_dir = normal_images_dir
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.shuffle = shuffle
        self.augment = augment
        
        # Carregar lista de sequências 360
        self.sequences_360 = []
        if os.path.exists(sequences_360_dir):
            for seq_folder in os.listdir(sequences_360_dir):
                seq_path = os.path.join(sequences_360_dir, seq_folder)
                if os.path.isdir(seq_path):
                    frames = sorted([f for f in os.listdir(seq_path) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if len(frames) >= sequence_length:
                        self.sequences_360.append(seq_path)
        
        # Carregar lista de imagens normais
        self.normal_images = []
        if os.path.exists(normal_images_dir):
            self.normal_images = [os.path.join(normal_images_dir, f) 
                                 for f in os.listdir(normal_images_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Criar índices balanceados
        self.samples = []
        # Adicionar sequências 360
        for seq_path in self.sequences_360:
            self.samples.append(('360', seq_path))
        
        # Adicionar imagens normais (repetir para balancear se necessário)
        num_normal_needed = len(self.sequences_360)
        if len(self.normal_images) > 0:
            normal_indices = np.random.choice(len(self.normal_images), 
                                             size=num_normal_needed, 
                                             replace=True)
            for idx in normal_indices:
                self.samples.append(('normal', self.normal_images[idx]))
        
        self.indexes = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        
        print(f"Dataset carregado:")
        print(f"  - Sequências 360: {len(self.sequences_360)}")
        print(f"  - Imagens normais: {len(self.normal_images)}")
        print(f"  - Total de amostras: {len(self.samples)}")
    
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
            
            if label == '360':
                # Carregar sequência de frames
                sequence = self._load_sequence(path)
            else:
                # Criar "sequência" de imagem normal repetida
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
        
        sequence = []
        for frame_name in selected_frames:
            frame_path = os.path.join(seq_path, frame_name)
            img = cv2.imread(frame_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            if self.augment:
                img = self._augment_image(img)
            
            img = img.astype(np.float32) / 255.0
            sequence.append(img)
        
        return np.array(sequence)
    
    def _load_normal_as_sequence(self, img_path):
        """Carrega uma imagem normal e cria uma sequência com pequenas variações."""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        sequence = []
        for i in range(self.sequence_length):
            frame = img.copy()
            
            if self.augment:
                # Aplicar pequenas variações para simular movimento natural
                frame = self._augment_image(frame)
            
            frame = frame.astype(np.float32) / 255.0
            sequence.append(frame)
        
        return np.array(sequence)
    
    def _augment_image(self, img):
        """Aplica data augmentation em uma imagem."""
        # Rotação aleatória
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h))
        
        # Flip horizontal
        if np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # Ajuste de brilho
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        return img
    
    def on_epoch_end(self):
        """Atualiza índices após cada época."""
        if self.shuffle:
            np.random.shuffle(self.indexes)
