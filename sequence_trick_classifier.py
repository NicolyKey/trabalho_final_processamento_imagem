import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
from sequence_data_generator import SequenceDataGenerator


class SequenceTrickClassifier:
    """Classificador de manobras baseado em sequências de frames usando CNN + LSTM."""
    
    def __init__(self, img_height=224, img_width=224, sequence_length=15, num_classes=2):
        self.img_height = img_height
        self.img_width = img_width
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['normal', '360']
        
    def build_cnn_lstm_model(self):
        """Constrói modelo CNN + LSTM para classificação de sequências."""
        
        # CNN base para extração de features de cada frame
        cnn_base = keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        cnn_base.trainable = False
        
        # Modelo para processar cada frame
        frame_input = layers.Input(shape=(self.img_height, self.img_width, 3))
        x = layers.Rescaling(1./127.5, offset=-1)(frame_input)
        x = cnn_base(x)
        x = layers.GlobalAveragePooling2D()(x)
        frame_features = keras.Model(inputs=frame_input, outputs=x)
        
        # Modelo completo para sequências
        sequence_input = layers.Input(shape=(self.sequence_length, self.img_height, self.img_width, 3))
        
        # Aplicar CNN em cada frame da sequência
        x = layers.TimeDistributed(frame_features)(sequence_input)
        
        # Camadas LSTM para capturar dependências temporais
        x = layers.LSTM(128, return_sequences=True, dropout=0.3)(x)
        x = layers.LSTM(64, dropout=0.3)(x)
        
        # Camadas densas finais
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs=sequence_input, outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_conv3d_model(self):
        """Constrói modelo com Conv3D para processar sequências diretamente."""
        
        self.model = keras.Sequential([
            layers.Input(shape=(self.sequence_length, self.img_height, self.img_width, 3)),
            
            # Primeira camada Conv3D
            layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.BatchNormalization(),
            
            # Segunda camada Conv3D
            layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.BatchNormalization(),
            
            # Terceira camada Conv3D
            layers.Conv3D(128, (3, 3, 3), activation='relu', padding='same'),
            layers.MaxPooling3D((2, 2, 2)),
            layers.BatchNormalization(),
            
            # Flatten e camadas densas
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, sequences_360_train, normal_images_train, 
              sequences_360_val=None, normal_images_val=None,
              epochs=50, batch_size=8, model_type='cnn_lstm'):
        """
        Treina o classificador de sequências.
        
        Args:
            sequences_360_train: Diretório com sequências de 360 para treino
            normal_images_train: Diretório com imagens normais para treino
            sequences_360_val: Diretório com sequências de 360 para validação
            normal_images_val: Diretório com imagens normais para validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
            model_type: 'cnn_lstm' ou 'conv3d'
        """
        
        print(f"Iniciando treinamento com modelo {model_type}...")
        print(f"Sequências 360 treino: {sequences_360_train}")
        print(f"Imagens normais treino: {normal_images_train}")
        
        # Construir modelo
        if model_type == 'cnn_lstm':
            self.build_cnn_lstm_model()
        elif model_type == 'conv3d':
            self.build_conv3d_model()
        else:
            raise ValueError(f"Tipo de modelo inválido: {model_type}")
        
        print(f"\nResumo do modelo:")
        self.model.summary()
        
        # Criar geradores de dados
        train_generator = SequenceDataGenerator(
            sequences_360_dir=sequences_360_train,
            normal_images_dir=normal_images_train,
            batch_size=batch_size,
            img_height=self.img_height,
            img_width=self.img_width,
            sequence_length=self.sequence_length,
            shuffle=True,
            augment=True
        )
        
        validation_generator = None
        if sequences_360_val and normal_images_val:
            validation_generator = SequenceDataGenerator(
                sequences_360_dir=sequences_360_val,
                normal_images_dir=normal_images_val,
                batch_size=batch_size,
                img_height=self.img_height,
                img_width=self.img_width,
                sequence_length=self.sequence_length,
                shuffle=False,
                augment=False
            )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_generator else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_generator else 'loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                'best_sequence_trick_model.h5',
                monitor='val_accuracy' if validation_generator else 'accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        # Treinar
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def save_model(self, filepath='sequence_trick_classifier_model.h5'):
        """Salva o modelo treinado."""
        if self.model:
            self.model.save(filepath)
            print(f"Modelo salvo em: {filepath}")
            
            # Salvar informações do modelo
            config = {
                'img_height': self.img_height,
                'img_width': self.img_width,
                'sequence_length': self.sequence_length,
                'num_classes': self.num_classes,
                'class_names': self.class_names
            }
            
            import json
            config_path = filepath.replace('.h5', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            # Salvar nomes das classes
            with open(filepath.replace('.h5', '_classes.txt'), 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
    
    def load_model(self, filepath='sequence_trick_classifier_model.h5'):
        """Carrega um modelo salvo."""
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
        
        # Carregar configuração
        import json
        config_path = filepath.replace('.h5', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.img_height = config['img_height']
                self.img_width = config['img_width']
                self.sequence_length = config['sequence_length']
                self.num_classes = config['num_classes']
                self.class_names = config['class_names']
        
        # Carregar nomes das classes
        classes_file = filepath.replace('.h5', '_classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
    
    def preprocess_sequence(self, image_sequence):
        """Preprocessa uma sequência de imagens para predição."""
        processed_sequence = []
        
        # Ajustar número de frames
        if len(image_sequence) > self.sequence_length:
            indices = np.linspace(0, len(image_sequence) - 1, self.sequence_length, dtype=int)
            selected_frames = [image_sequence[i] for i in indices]
        else:
            selected_frames = image_sequence[:self.sequence_length]
            while len(selected_frames) < self.sequence_length:
                selected_frames.append(image_sequence[-1])
        
        for img in selected_frames:
            # Converter para RGB se necessário
            if isinstance(img, str):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # Assumir que já está em RGB
                pass
            
            # Redimensionar
            img = cv2.resize(img, (self.img_width, self.img_height))
            
            # Normalizar
            img = img.astype(np.float32) / 255.0
            processed_sequence.append(img)
        
        # Adicionar dimensão do batch
        processed_sequence = np.array(processed_sequence)
        processed_sequence = np.expand_dims(processed_sequence, axis=0)
        
        return processed_sequence
    
    def predict_sequence(self, image_sequence):
        """Prediz a classe de uma sequência de imagens."""
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() ou train() primeiro.")
        
        processed_sequence = self.preprocess_sequence(image_sequence)
        predictions = self.model.predict(processed_sequence, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': {self.class_names[i]: float(predictions[0][i]) 
                               for i in range(len(self.class_names))}
        }


if __name__ == "__main__":
    print("Classificador de Sequências de Manobras")
    print("=" * 50)
    print("\nPara treinar o modelo, organize seus dados em:")
    print("  - sequences_dataset/360/ (subpastas com sequências de frames)")
    print("  - dataset/train/normal/ (imagens de bikes normais)")
    print("\nExemplo de uso:")
    print("  classifier = SequenceTrickClassifier()")
    print("  classifier.train(")
    print("      sequences_360_train='sequences_dataset/360',")
    print("      normal_images_train='dataset/train/normal'")
    print("  )")
