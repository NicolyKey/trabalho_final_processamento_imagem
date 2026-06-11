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
        
        base_model.trainable = False
        
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
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
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
    
    def train(self, train_dir, validation_dir=None, epochs=50, batch_size=32, use_transfer_learning=True):
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
        
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
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
            val_datagen = ImageDataGenerator(rescale=1./255)
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
                callbacks=callbacks
            )
        else:
            history = self.model.fit(
                train_generator,
                epochs=epochs,
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
    
    def predict_sequence(self, image_sequence, aggregate='max'):
        predictions = []
        
        for image in image_sequence:
            pred = self.predict(image)
            predictions.append(pred)
        
        if aggregate == 'max':
            trick_360_confidences = [p['all_predictions']['360'] for p in predictions]
            max_idx = np.argmax(trick_360_confidences)
            return predictions[max_idx]
        
        elif aggregate == 'average':
            avg_predictions = {}
            for class_name in self.class_names:
                avg_predictions[class_name] = np.mean([p['all_predictions'][class_name] 
                                                       for p in predictions])
            
            predicted_class = max(avg_predictions, key=avg_predictions.get)
            return {
                'class': predicted_class,
                'confidence': avg_predictions[predicted_class],
                'all_predictions': avg_predictions
            }
        
        return predictions


if __name__ == "__main__":
    classifier = TrickClassifier()
    
    print("Para treinar o modelo, organize suas imagens em:")
    print("  dataset/train/normal/")
    print("  dataset/train/360/")
    print("  dataset/validation/normal/")
    print("  dataset/validation/360/")
    print("\nDepois execute:")
    print("  classifier.train('dataset/train', 'dataset/validation')")
