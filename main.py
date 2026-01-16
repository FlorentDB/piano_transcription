#!/usr/bin/env python3
"""
Script principal pour prétraiter les données et lancer l'entraînement
"""

import argparse
import sys
from pathlib import Path


def preprocess_datasets(args):
    """Prétraite les datasets bruts"""
    from src.model.preprocess_datasets import DatasetPreprocessor
    
    print("\n🔄 Starting dataset preprocessing...")
    
    preprocessor = DatasetPreprocessor(
        source_dir=args.source_dir,
        output_dir=args.output_dir
    )
    
    preprocessor.preprocess_all()
    
    print("\n✅ Preprocessing completed!")
    print(f"📁 Preprocessed data saved in: {args.output_dir}")


def train_model(args):
    """Lance l'entraînement"""
    import torch
    from src.model.dataset_loader import create_dataloaders
    from src.model.training import Trainer
    
    # Import du modèle (à adapter selon votre implémentation)
    try:
        from src.model.model import PARModel, PARCompactModel
    except ImportError:
        print(" Error: Could not import model.")
        print("Make sure you have the model implementation in src/model/model.py")
        sys.exit(1)
    
    print("\n Starting training...")
    print("="*60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Créer les dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=args.data_dir,
        datasets=args.datasets,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length
    )
    
    # Créer le modèle
    print("\nInitializing model...")
    if args.model_type == 'par':
        model = PARModel()
    elif args.model_type == 'compact':
        model = PARCompactModel()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Créer le trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Charger un checkpoint si spécifié
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Lancer l'entraînement
    trainer.train()

    print("\n Training completed!")


def evaluate_model(args):
    """Évalue un modèle entraîné"""
    print("\nStarting evaluation...")
    print("Not implemented yet - TODO")
    # TODO: Implémenter l'évaluation avec les métriques de l'article


def main():
    parser = argparse.ArgumentParser(
        description="Piano Transcription - Preprocessing and Training"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Commande: preprocess
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Preprocess raw datasets'
    )
    preprocess_parser.add_argument(
        '--source-dir',
        type=str,
        default='/home/spycologue/dataset',
        help='Path to raw datasets'
    )
    preprocess_parser.add_argument(
        '--output-dir',
        type=str,
        default='./dataset',
        help='Path to save preprocessed data'
    )
    
    # Commande: train
    train_parser = subparsers.add_parser(
        'train',
        help='Train the model'
    )
    train_parser.add_argument(
        '--data-dir',
        type=str,
        default='./dataset',
        help='Path to preprocessed data'
    )
    train_parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=['maestro'],
        choices=['maestro', 'smd'],
        help='Datasets to use for training'
    )
    train_parser.add_argument(
        '--model-type',
        type=str,
        default='par',
        choices=['par', 'compact'],
        help='Model type: par (19.7M params) or compact (2.7M params)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=12,
        help='Batch size'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--max-iterations',
        type=int,
        default=250000,
        help='Maximum training iterations'
    )
    train_parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    train_parser.add_argument(
        '--sequence-length',
        type=int,
        default=312,
        help='Sequence length in frames (~10s at 31.25 fps)'
    )
    train_parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    # Commande: evaluate
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained model'
    )
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    eval_parser.add_argument(
        '--data-dir',
        type=str,
        default='./dataset',
        help='Path to preprocessed data'
    )
    eval_parser.add_argument(
        '--dataset',
        type=str,
        default='maestro',
        choices=['maestro', 'smd'],
        help='Dataset to evaluate on'
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Exécuter la commande
    if args.command == 'preprocess':
        preprocess_datasets(args)
    elif args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)


if __name__ == "__main__":
    main()