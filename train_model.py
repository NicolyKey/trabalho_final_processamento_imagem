import os
import argparse
from trick_classifier import TrickClassifier


def create_dataset_structure():
    directories = [
        'dataset/train/normal',
        'dataset/train/360',
        'dataset/validation/normal',
        'dataset/validation/360'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Criado: {directory}")
    
    print("\nEstrutura de diretórios criada!")
    print("\nAgora você precisa:")
    print("1. Colocar imagens de bicicletas em posição normal em dataset/train/normal/")
    print("2. Colocar imagens de bicicletas fazendo 360 em dataset/train/360/")
    print("3. Colocar imagens de validação em dataset/validation/normal/ e dataset/validation/360/")


def train_classifier(train_dir, val_dir=None, epochs=50, batch_size=32, use_transfer_learning=True):
    print("Iniciando treinamento do classificador de manobras...")
    print(f"Diretório de treinamento: {train_dir}")
    print(f"Diretório de validação: {val_dir}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Transfer Learning: {use_transfer_learning}")
    
    classifier = TrickClassifier(img_height=224, img_width=224, num_classes=2)

    class_weight = {0: 1.0, 1: 2.0}  # ← adiciona aqui

    history = classifier.train(
        train_dir=train_dir,
        validation_dir=val_dir,
        epochs=epochs,
        batch_size=batch_size,
        use_transfer_learning=use_transfer_learning,
        class_weight=class_weight 
    )
    
    classifier.save_model('trick_classifier_model.h5')
    
    print("\nTreinamento concluído!")
    print("Modelo salvo em: trick_classifier_model.h5")
    
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treinar classificador de manobras de bicicleta')
    parser.add_argument('--setup', action='store_true', 
                       help='Criar estrutura de diretórios para o dataset')
    parser.add_argument('--train_dir', type=str, default='dataset/train',
                       help='Diretório com dados de treinamento')
    parser.add_argument('--val_dir', type=str, default='dataset/validation',
                       help='Diretório com dados de validação')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas de treinamento')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--no_transfer_learning', action='store_true',
                       help='Não usar transfer learning (treinar CNN do zero)')
    
    args = parser.parse_args()
    
    if args.setup:
        create_dataset_structure()
    else:
        if not os.path.exists(args.train_dir):
            print(f"Erro: Diretório de treinamento não encontrado: {args.train_dir}")
            print("Execute com --setup para criar a estrutura de diretórios")
        else:
            val_dir = args.val_dir if os.path.exists(args.val_dir) else None
            
            train_classifier(
                train_dir=args.train_dir,
                val_dir=val_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                use_transfer_learning=not args.no_transfer_learning
            )
