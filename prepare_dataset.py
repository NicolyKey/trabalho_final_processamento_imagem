import os
import shutil
from pathlib import Path


def check_dataset_structure(base_dir='dataset'):
    print("="*60)
    print("VERIFICAÇÃO DA ESTRUTURA DO DATASET")
    print("="*60)
    
    required_dirs = [
        'dataset/train/normal',
        'dataset/train/360',
        'dataset/validation/normal',
        'dataset/validation/360'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        status = "✓" if exists else "✗"
        print(f"{status} {dir_path}")
        
        if exists:
            images = [f for f in os.listdir(dir_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  → {len(images)} imagens encontradas")
        else:
            all_exist = False
    
    print("="*60)
    
    if not all_exist:
        print("\n⚠️  ATENÇÃO: Estrutura de diretórios incompleta!")
        print("Execute: python train_model.py --setup")
        return False
    
    train_normal = len([f for f in os.listdir('dataset/train/normal') 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    train_360 = len([f for f in os.listdir('dataset/train/360') 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if train_normal == 0 or train_360 == 0:
        print("\n⚠️  ERRO: Você precisa adicionar imagens nas pastas!")
        print("\nPasso a passo:")
        print("1. Execute: python main.py --mode extract")
        print("2. Isso criará a pasta 'bike_frames/' com imagens de bicicletas")
        print("3. Separe manualmente as imagens:")
        print("   - Copie imagens normais para dataset/train/normal/")
        print("   - Copie imagens com 360 para dataset/train/360/")
        print("4. Faça o mesmo para dataset/validation/")
        return False
    
    print("\n✓ Dataset pronto para treinamento!")
    print(f"  - Treinamento: {train_normal + train_360} imagens")
    print(f"    • Normal: {train_normal}")
    print(f"    • 360: {train_360}")
    
    return True


def organize_extracted_frames(source_dir='bike_frames', train_split=0.8):
    if not os.path.exists(source_dir):
        print(f"Erro: Pasta {source_dir} não encontrada!")
        print("Execute primeiro: python main.py --mode extract")
        return
    
    images = [f for f in os.listdir(source_dir) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(images) == 0:
        print(f"Nenhuma imagem encontrada em {source_dir}")
        return
    
    print(f"\nEncontradas {len(images)} imagens em {source_dir}")
    print("\n⚠️  ATENÇÃO: Este script não pode classificar automaticamente!")
    print("Você precisa separar manualmente as imagens em:")
    print("  - dataset/train/normal/")
    print("  - dataset/train/360/")
    print("  - dataset/validation/normal/")
    print("  - dataset/validation/360/")
    print("\nRecomendação:")
    print("1. Abra a pasta bike_frames/")
    print("2. Visualize cada imagem")
    print("3. Copie para a pasta apropriada")


def create_sample_dataset_from_video():
    print("\n" + "="*60)
    print("CRIANDO DATASET DE EXEMPLO")
    print("="*60)
    
    print("\nPasso 1: Extraindo frames do vídeo...")
    print("Execute: python main.py --mode extract --video videos/crianca_bicicleta.mp4")
    print("\nPasso 2: Organize as imagens manualmente")
    print("Passo 3: Execute o treinamento")
    print("="*60)


def quick_test_model():
    import cv2
    import numpy as np
    from trick_classifier import TrickClassifier
    
    print("\n" + "="*60)
    print("TESTE RÁPIDO DO MODELO")
    print("="*60)
    
    if not os.path.exists('trick_classifier_model.h5'):
        print("Modelo não encontrado! Treine primeiro com:")
        print("python train_model.py --train_dir dataset/train")
        return
    
    classifier = TrickClassifier()
    classifier.load_model('trick_classifier_model.h5')
    
    print(f"Modelo carregado com sucesso!")
    print(f"Classes: {classifier.class_names}")
    
    test_dir = 'bike_frames'
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]
        
        print(f"\nTestando com {len(images)} imagens de exemplo:")
        for img_name in images:
            img_path = os.path.join(test_dir, img_name)
            result = classifier.predict(img_path)
            print(f"  {img_name}: {result['class']} ({result['confidence']:.2%})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preparar dataset para treinamento')
    parser.add_argument('--check', action='store_true', 
                       help='Verificar estrutura do dataset')
    parser.add_argument('--organize', action='store_true',
                       help='Ajudar a organizar frames extraídos')
    parser.add_argument('--guide', action='store_true',
                       help='Mostrar guia completo')
    parser.add_argument('--test', action='store_true',
                       help='Testar modelo treinado')
    
    args = parser.parse_args()
    
    if args.check:
        check_dataset_structure()
    elif args.organize:
        organize_extracted_frames()
    elif args.test:
        quick_test_model()
    elif args.guide:
        create_sample_dataset_from_video()
    else:
        print("Uso: python prepare_dataset.py [--check|--organize|--guide|--test]")
        print("\nOpções:")
        print("  --check     Verificar estrutura do dataset")
        print("  --organize  Ajudar a organizar frames extraídos")
        print("  --guide     Mostrar guia completo de preparação")
        print("  --test      Testar modelo treinado")
