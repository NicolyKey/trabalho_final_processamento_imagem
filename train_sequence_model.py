import os
import argparse
from sequence_trick_classifier import SequenceTrickClassifier


def train_sequence_classifier(sequences_360_train, normal_images_train,
                              sequences_360_val=None, normal_images_val=None,
                              epochs=50, batch_size=8, model_type='cnn_lstm',
                              sequence_length=15):
    """
    Treina o classificador de manobras baseado em sequências.
    
    Args:
        sequences_360_train: Diretório com sequências de 360 para treino
        normal_images_train: Diretório com imagens normais para treino
        sequences_360_val: Diretório com sequências de 360 para validação
        normal_images_val: Diretório com imagens normais para validação
        epochs: Número de épocas
        batch_size: Tamanho do batch
        model_type: 'cnn_lstm' ou 'conv3d'
        sequence_length: Número de frames por sequência
    """
    
    print("=" * 70)
    print("TREINAMENTO DE CLASSIFICADOR DE MANOBRAS BASEADO EM SEQUÊNCIAS")
    print("=" * 70)
    print(f"\nConfiguração:")
    print(f"  - Sequências 360 (treino): {sequences_360_train}")
    print(f"  - Imagens normais (treino): {normal_images_train}")
    print(f"  - Sequências 360 (validação): {sequences_360_val}")
    print(f"  - Imagens normais (validação): {normal_images_val}")
    print(f"  - Épocas: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Tipo de modelo: {model_type}")
    print(f"  - Comprimento da sequência: {sequence_length} frames")
    print()
    
    # Verificar se os diretórios existem
    if not os.path.exists(sequences_360_train):
        raise ValueError(f"Diretório de sequências 360 não encontrado: {sequences_360_train}")
    
    if not os.path.exists(normal_images_train):
        raise ValueError(f"Diretório de imagens normais não encontrado: {normal_images_train}")
    
    # Criar classificador
    classifier = SequenceTrickClassifier(
        img_height=224,
        img_width=224,
        sequence_length=sequence_length,
        num_classes=2
    )
    
    # Treinar
    history = classifier.train(
        sequences_360_train=sequences_360_train,
        normal_images_train=normal_images_train,
        sequences_360_val=sequences_360_val,
        normal_images_val=normal_images_val,
        epochs=epochs,
        batch_size=batch_size,
        model_type=model_type
    )
    
    # Salvar modelo
    model_filename = f'sequence_trick_classifier_{model_type}.h5'
    classifier.save_model(model_filename)
    
    print("\n" + "=" * 70)
    print("TREINAMENTO CONCLUÍDO!")
    print("=" * 70)
    print(f"Modelo salvo em: {model_filename}")
    print(f"Melhor modelo salvo em: best_sequence_trick_model.h5")
    
    return history, classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Treinar classificador de manobras baseado em sequências de frames'
    )
    
    # Diretórios de dados
    parser.add_argument('--sequences_360_train', type=str, 
                       default='sequences_dataset/360',
                       help='Diretório com sequências de 360 para treinamento')
    
    parser.add_argument('--normal_train', type=str,
                       default='dataset/train/normal',
                       help='Diretório com imagens normais para treinamento')
    
    parser.add_argument('--sequences_360_val', type=str,
                       default=None,
                       help='Diretório com sequências de 360 para validação')
    
    parser.add_argument('--normal_val', type=str,
                       default='dataset/validation/normal',
                       help='Diretório com imagens normais para validação')
    
    # Hiperparâmetros
    parser.add_argument('--epochs', type=int, default=50,
                       help='Número de épocas de treinamento')
    
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Tamanho do batch')
    
    parser.add_argument('--sequence_length', type=int, default=15,
                       help='Número de frames por sequência')
    
    parser.add_argument('--model_type', type=str, default='cnn_lstm',
                       choices=['cnn_lstm', 'conv3d'],
                       help='Tipo de modelo: cnn_lstm ou conv3d')
    
    args = parser.parse_args()
    
    # Verificar se diretório de validação existe
    sequences_360_val = args.sequences_360_val if args.sequences_360_val and os.path.exists(args.sequences_360_val) else None
    normal_val = args.normal_val if os.path.exists(args.normal_val) else None
    
    if sequences_360_val is None and normal_val is None:
        print("\nAVISO: Nenhum conjunto de validação encontrado.")
        print("O treinamento será feito sem validação.")
        print()
    
    # Treinar
    train_sequence_classifier(
        sequences_360_train=args.sequences_360_train,
        normal_images_train=args.normal_train,
        sequences_360_val=sequences_360_val,
        normal_images_val=normal_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model_type,
        sequence_length=args.sequence_length
    )
