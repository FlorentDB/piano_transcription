import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

from src.model.model import PARModel
from dataset import create_dataloaders  # Le fichier dataset.py que j'ai créé


class FocalLoss(nn.Module):
    """Focal Loss selon l'article (alpha=1.0, gamma=2.0)"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, 88, 5, T) - model output logits
            targets: (B, 88, T) - ground truth note states (0-4)
        Returns:
            loss: scalar loss value
        """
        B, num_pitches, num_classes, T = inputs.shape
        
        # Reshape pour CrossEntropyLoss
        inputs = inputs.permute(0, 2, 1, 3)  # (B, 5, 88, T)
        inputs = inputs.reshape(B * T * num_pitches, num_classes)  # (B*T*88, 5)
        targets = targets.reshape(B * T * num_pitches)  # (B*T*88,)
        
        # Calculer Cross Entropy
        ce_loss = self.ce_loss(inputs, targets)
        
        # Calculer la probabilité de la vraie classe
        p = torch.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Appliquer focal loss
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss
        
        return loss.mean()


class VelocityLoss(nn.Module):
    """L2 Loss pour la vélocité (seulement sur les frames d'onset)"""
    
    def __init__(self):
        super(VelocityLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred_velocities, true_velocities, states):
        """
        Args:
            pred_velocities: (B, 88, T) - prédictions de vélocité [0, 1]
            true_velocities: (B, 88, T) - vélocités réelles [0, 1]
            states: (B, 88, T) - états pour identifier les onsets
        Returns:
            loss: scalar
        """
        # Masque pour les onsets uniquement (state == 1 ou 3)
        onset_mask = ((states == 1) | (states == 3)).float()
        
        # Calculer MSE seulement sur les onsets
        loss = self.mse(pred_velocities, true_velocities)
        loss = (loss * onset_mask).sum() / (onset_mask.sum() + 1e-8)
        
        return loss


class Trainer:
    """Classe pour gérer l'entraînement"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-3,
        max_iterations=250000,
        checkpoint_dir='checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.max_iterations = max_iterations
        
        # Losses
        self.note_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        self.velocity_loss_fn = VelocityLoss()
        
        # Optimizer (AdaBelief selon l'article, mais Adam fonctionne aussi)
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Tracking
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_val_loss = float('inf')
        self.iteration = 0
        
    def train_epoch(self):
        """Entraîne pendant une époque"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        
        for mel_spec, state_labels, context in pbar:
            mel_spec = mel_spec.to(self.device)
            state_labels = state_labels.to(self.device)
            context = context.to(self.device)
            
            # Forward pass
            state_probs, velocities = self.model(mel_spec, context)
            
            # Calculer loss (note states seulement pour l'instant)
            loss = self.note_loss_fn(state_probs, state_labels)
            
            # TODO: Ajouter velocity loss si vous avez implémenté la prédiction
            # vel_loss = self.velocity_loss_fn(velocities, true_velocities, state_labels)
            # loss = loss + vel_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (bonne pratique)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            # Tracking
            total_loss += loss.item()
            num_batches += 1
            self.iteration += 1
            
            pbar.set_postfix({'loss': loss.item(), 'iter': self.iteration})
            
            # Validation périodique
            if self.iteration % 1000 == 0:
                val_loss = self.validate()
                self.model.train()
                
                # Sauvegarder si meilleur modèle
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                
            # Arrêter si max iterations atteint
            if self.iteration >= self.max_iterations:
                break
        
        return total_loss / num_batches
    
    def validate(self):
        """Valide le modèle"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for mel_spec, state_labels, context in tqdm(self.val_loader, desc='Validation'):
                mel_spec = mel_spec.to(self.device)
                state_labels = state_labels.to(self.device)
                context = context.to(self.device)
                
                # Forward pass
                state_probs, velocities = self.model(mel_spec, context)
                
                # Calculer loss
                loss = self.note_loss_fn(state_probs, state_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"\nValidation Loss: {avg_loss:.4f} (Best: {self.best_val_loss:.4f})")
        
        return avg_loss
    
    def save_checkpoint(self, filename):
        """Sauvegarde un checkpoint"""
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename):
        """Charge un checkpoint"""
        path = self.checkpoint_dir / filename
        
        if not path.exists():
            print(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded: {path} (iteration {self.iteration})")
    
    def train(self):
        """Boucle d'entraînement principale"""
        print(f"Starting training for {self.max_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        epoch = 0
        while self.iteration < self.max_iterations:
            epoch += 1
            print(f"\n=== Epoch {epoch} ===")
            
            avg_loss = self.train_epoch()
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Sauvegarder périodiquement
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
            
            if self.iteration >= self.max_iterations:
                break
        
        print("\nTraining completed!")
        self.save_checkpoint('final_model.pt')


def main():
    """Fonction principale d'entraînement"""
    
    # Configuration
    config = {
        'data_dir': '/path/to/maestro-v3.0.0',  # À MODIFIER
        'dataset_name': 'maestro',  # 'maestro', 'maps', ou 'smd'
        'batch_size': 12,
        'learning_rate': 1e-3,
        'max_iterations': 250000,
        'num_workers': 4,
        'sequence_length': 100,  # 10s à 31.25 fps ≈ 312 frames, mais commencez petit
        'checkpoint_dir': 'checkpoints',
    }
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Créer les dataloaders
    print("\nLoading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        dataset_name=config['dataset_name'],
        sequence_length=config['sequence_length']
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Créer le modèle
    print("\nInitializing model...")
    model = PARModel()  # Ou PARCompactModel pour la version compacte
    
    # Créer le trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config['learning_rate'],
        max_iterations=config['max_iterations'],
        checkpoint_dir=config['checkpoint_dir']
    )
    
    # Charger un checkpoint si vous voulez reprendre l'entraînement
    # trainer.load_checkpoint('checkpoint_epoch_10.pt')
    
    # Lancer l'entraînement
    trainer.train()


if __name__ == "__main__":
    main()