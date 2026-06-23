import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import os


class TrickClassifier:
    def __init__(self, img_height=224, img_width=224, num_classes=2):
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['normal', '360']
        
    def build_model(self):
        base_model = keras.applications.MobileNetV2(
            input_shape=(self.img_height, self.img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        self.base_model = base_model
        
        self.model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            layers.Rescaling(1./127.5, offset=-1),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def build_cnn_model(self):
        self.model = keras.Sequential([
            layers.Input(shape=(self.img_height, self.img_width, 3)),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.BatchNormalization(),
            
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return self.model
    
    def train(self, train_dir, validation_dir=None, epochs=50, batch_size=32, use_transfer_learning=True, class_weight=None):
        if not os.path.exists(train_dir):
            raise ValueError(f"Diretório de treinamento não encontrado: {train_dir}")
        
        subdirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
        if len(subdirs) < 2:
            raise ValueError(f"O diretório {train_dir} deve conter pelo menos 2 subpastas (classes)")
        
        total_images = 0
        for subdir in subdirs:
            subdir_path = os.path.join(train_dir, subdir)
            images = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(images)
            print(f"Classe '{subdir}': {len(images)} imagens")
        
        if total_images == 0:
            raise ValueError(f"Nenhuma imagem encontrada em {train_dir}")
        
        if use_transfer_learning:
            self.build_model()
        else:
            self.build_cnn_model()
        
        rescale_value = None if use_transfer_learning else 1./255
        train_datagen = ImageDataGenerator(
            rescale=rescale_value,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.15,
            zoom_range=0.3,
            brightness_range=[0.7, 1.3],
            channel_shift_range=15,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_height, self.img_width),
            batch_size=batch_size,
            class_mode='categorical'
        )
        
        self.class_names = list(train_generator.class_indices.keys())
        print(f"\nClasses detectadas: {self.class_names}")
        print(f"Total de imagens de treinamento: {train_generator.samples}")
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_dir else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_dir else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                'best_trick_model.h5',
                monitor='val_accuracy' if validation_dir else 'accuracy',
                save_best_only=True,
                mode='max'
            )
        ]
        
        if validation_dir:
            val_datagen = ImageDataGenerator(rescale=rescale_value)
            validation_generator = val_datagen.flow_from_directory(
                validation_dir,
                target_size=(self.img_height, self.img_width),
                batch_size=batch_size,
                class_mode='categorical'
            )
            
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                validation_data=validation_generator,
                class_weight=class_weight,
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
                class_weight=class_weight,
                callbacks=callbacks
            )
        
        return history
    
    def save_model(self, filepath='trick_classifier_model.h5'):
        if self.model:
            self.model.save(filepath)
            print(f"Modelo salvo em: {filepath}")
            
            with open(filepath.replace('.h5', '_classes.txt'), 'w') as f:
                for class_name in self.class_names:
                    f.write(f"{class_name}\n")
    
    def load_model(self, filepath='trick_classifier_model.h5'):
        self.model = keras.models.load_model(filepath)
        print(f"Modelo carregado de: {filepath}")
        
        classes_file = filepath.replace('.h5', '_classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
    
    def preprocess_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image.dtype != np.uint8:
            if image.dtype == np.float32 or image.dtype == np.float64:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        image = cv2.resize(image, (self.img_width, self.img_height))
        
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        if self.model is None:
            raise ValueError("Modelo não carregado. Use load_model() ou train() primeiro.")
        
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image, verbose=0)
        
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_names[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(confidence),
            'all_predictions': {self.class_names[i]: float(predictions[0][i]) 
                               for i in range(len(self.class_names))}
        }
    
    def compute_visual_change(self, image_sequence):
        """Mede a mudança visual ao longo da sequência de frames.
        
        Um 360 causa alta mudança visual (a bike mostra diferentes ângulos).
        Um empinando mantém aparência estável (mesma orientação).
        """
        if len(image_sequence) < 5:
            return 0.0, 0.0, 0.0
        
        resized = []
        for img in image_sequence:
            r = cv2.resize(img, (64, 64))
            if len(r.shape) == 3:
                r = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
            resized.append(r.astype(np.float32))
        
        frame_diffs = []
        for i in range(1, len(resized)):
            diff = np.mean(np.abs(resized[i] - resized[i-1])) / 255.0
            frame_diffs.append(diff)
        
        quarter = max(1, len(resized) // 4)
        long_diffs = []
        for i in range(0, len(resized) - quarter, quarter):
            diff = np.mean(np.abs(resized[i] - resized[i + quarter])) / 255.0
            long_diffs.append(diff)
        
        start_end_diff = np.mean(np.abs(resized[0] - resized[-1])) / 255.0
        
        avg_motion = np.mean(frame_diffs) if frame_diffs else 0.0
        avg_long_change = np.mean(long_diffs) if long_diffs else 0.0
        
        return avg_motion, avg_long_change, start_end_diff
    
    def compute_histogram_variance(self, image_sequence):
        """Mede variação nos histogramas ao longo da sequência.
        
        360: histogramas mudam muito (diferentes partes da bike visíveis).
        Estável: histogramas permanecem similares.
        """
        if len(image_sequence) < 5:
            return 0.0
        
        histograms = []
        step = max(1, len(image_sequence) // 10)
        
        for i in range(0, len(image_sequence), step):
            img = cv2.resize(image_sequence[i], (64, 64))
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([img], [0], None, [32], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-7)
            histograms.append(hist)
        
        correlations = []
        for i in range(1, len(histograms)):
            corr = cv2.compareHist(
                histograms[i-1].astype(np.float32),
                histograms[i].astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        avg_corr = np.mean(correlations)
        variance_score = 1.0 - max(0.0, avg_corr)
        
        return variance_score
    
    def detect_360_motion(self, image_sequence, motion_threshold=0.08, change_threshold=0.12):
        """Detecta se a sequência contém movimento de 360 baseado em mudança visual.
        
        Retorna:
            - motion_score: pontuação combinada (0.0 a 1.0)
            - metrics: dicionário com métricas detalhadas
        """
        avg_motion, avg_long_change, start_end_diff = self.compute_visual_change(image_sequence)
        hist_variance = self.compute_histogram_variance(image_sequence)
        
        motion_score = min(1.0, avg_motion / motion_threshold)
        change_score = min(1.0, avg_long_change / change_threshold)
        
        combined = (motion_score * 0.3) + (change_score * 0.35) + (hist_variance * 0.35)
        
        metrics = {
            'avg_motion': avg_motion,
            'avg_long_change': avg_long_change,
            'start_end_diff': start_end_diff,
            'hist_variance': hist_variance,
            'motion_score': motion_score,
            'change_score': change_score
        }
        
        return combined, metrics
    
    def predict_sequence(self, image_sequence, aggregate='average'):
        """Classifica uma sequência combinando CNN + análise temporal de mudança visual."""
        predictions = []
        
        for image in image_sequence:
            pred = self.predict(image)
            predictions.append(pred)
        
        motion_score, metrics = self.detect_360_motion(image_sequence)
        
        if aggregate == 'max':
            trick_360_confidences = [p['all_predictions']['360'] for p in predictions]
            max_idx = np.argmax(trick_360_confidences)
            cnn_result = predictions[max_idx]
        elif aggregate == 'average':
            avg_predictions = {}
            for class_name in self.class_names:
                avg_predictions[class_name] = np.mean([p['all_predictions'][class_name] 
                                                       for p in predictions])
            predicted_class = max(avg_predictions, key=avg_predictions.get)
            cnn_result = {
                'class': predicted_class,
                'confidence': avg_predictions[predicted_class],
                'all_predictions': avg_predictions
            }
        else:
            cnn_result = predictions[-1]
        
        cnn_360_conf = cnn_result['all_predictions'].get('360', 0.0)
        
        combined_360 = (cnn_360_conf * 0.65) + (motion_score * 0.35)
        combined_normal = 1.0 - combined_360
        
        if combined_360 > 0.5:
            final_class = '360'
            final_confidence = combined_360
        else:
            final_class = 'normal'
            final_confidence = combined_normal
        
        return {
            'class': final_class,
            'confidence': float(final_confidence),
            'all_predictions': {
                '360': float(combined_360),
                'normal': float(combined_normal)
            },
            'motion_score': float(motion_score),
            'cnn_confidence': float(cnn_360_conf),
            'metrics': metrics
        }


if __name__ == "__main__":
    classifier = TrickClassifier()
    
    print("Para treinar o modelo, organize suas imagens em:")
    print("  dataset/train/normal/")
    print("  dataset/train/360/")
    print("  dataset/validation/normal/")
    print("  dataset/validation/360/")
    print("\nDepois execute:")
    print("  classifier.train('dataset/train', 'dataset/validation')")
