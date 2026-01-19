import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import h5py
from pathlib import Path
from tqdm import tqdm
import os
import signal
from model.model import PARModel
from torch.cuda.amp import GradScaler, autocast  # NEW: For mixed precision

# --- H5 Dataset Logic (unchanged) ---
H5_DEFAULT = "dataset/preprocessed_dataset.h5"
TRAIN_SET = 0.8
VAL_SET = 0.1
TEST_SET = 0.1
RNG_SEED = 42
PAD_DB = -80.0

def _list_all_chunk_keys(h5_path: str):
    keys = []
    with h5py.File(h5_path, "r") as f:
        for dataset_name in f.keys():
            root = f[dataset_name]
            for piece in root.keys():
                piece_group = root[piece]
                for chunk in piece_group.keys():
                    chunk_key = f"{dataset_name}/{piece}/{chunk}"
                    if 'spectrogram' in piece_group[chunk].keys() and 'multi_state' in piece_group[chunk].keys():
                        keys.append(chunk_key)
    return keys

def _split_keys_random(keys, train_set=TRAIN_SET, val_set=VAL_SET, test_set=TEST_SET, seed=RNG_SEED):
    assert abs(train_set + val_set + test_set - 1.0) < 1e-6
    rng = random.Random(seed)
    keys = keys.copy()
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * train_set)
    n_val = int(n * val_set)
    train = keys[:n_train]
    val = keys[n_train:n_train + n_val]
    test = keys[n_train + n_val:]
    return train, val, test

class H5ChunkDataset(Dataset):
    def __init__(self, h5_path: str, keys: list, sequence_length: int = 312, split: str = "train"):
        self.h5_path = h5_path
        self.keys = keys
        self.sequence_length = sequence_length
        self.split = split
        self._h5 = None

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_open()
        key = self.keys[idx]
        grp = self._h5[key]

        spec_np = np.array(grp['spectrogram'], dtype=np.float32)
        multi_np = np.array(grp['multi_state'], dtype=np.uint8)

        T_chunk, n_mels = spec_np.shape
        seq_len = self.sequence_length

        if T_chunk >= seq_len:
            start = random.randint(0, T_chunk - seq_len) if self.split == "train" else (T_chunk - seq_len) // 2
            end = start + seq_len
            spec_slice = spec_np[start:end, :]
            state_slice = multi_np[start:end, :]
        else:
            spec_slice = spec_np
            state_slice = multi_np

        spec_t = torch.from_numpy(spec_slice.T).float().unsqueeze(0)
        state_t = torch.from_numpy(state_slice.T).long()

        if spec_t.shape[-1] < seq_len:
            pad_len = seq_len - spec_t.shape[-1]
            pad_spec = torch.full((1, n_mels, pad_len), PAD_DB, dtype=spec_t.dtype)
            spec_t = torch.cat([spec_t, pad_spec], dim=2)
            pad_states = torch.zeros((88, pad_len), dtype=state_t.dtype)
            state_t = torch.cat([state_t, pad_states], dim=1)

        context = torch.zeros((88, 3), dtype=torch.float32)
        return spec_t, state_t, context

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,  # INCREASED from 4
    num_workers: int = 8,  # INCREASED for better data loading
    sequence_length: int = 312,
):
    h5_path = data_dir if data_dir is not None else H5_DEFAULT
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    keys = _list_all_chunk_keys(h5_path)
    if len(keys) == 0:
        raise RuntimeError(f"No chunks found in {h5_path}")

    train_keys, val_keys, test_keys = _split_keys_random(keys)

    def _worker_init_fn(worker_id):
        seed_worker = RNG_SEED + worker_id
        random.seed(seed_worker)
        np.random.seed(seed_worker)

    train_ds = H5ChunkDataset(h5_path, train_keys, sequence_length=sequence_length, split='train')
    val_ds = H5ChunkDataset(h5_path, val_keys, sequence_length=sequence_length, split='val')
    test_ds = H5ChunkDataset(h5_path, test_keys, sequence_length=sequence_length, split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn, persistent_workers=True)  # ADDED persistent_workers
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=_worker_init_fn)

    return train_loader, val_loader, test_loader

# --- Loss Functions (unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        B, num_pitches, num_classes, T = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3).reshape(B * T * num_pitches, num_classes)
        targets = targets.reshape(B * T * num_pitches)
        ce_loss = self.ce_loss(inputs, targets)
        p = torch.softmax(inputs, dim=-1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_weight * ce_loss
        return loss.mean()

class VelocityLoss(nn.Module):
    def __init__(self):
        super(VelocityLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_velocities, true_velocities, states):
        onset_mask = ((states == 1) | (states == 3)).float()
        loss = self.mse(pred_velocities, true_velocities)
        loss = (loss * onset_mask).sum() / (onset_mask.sum() + 1e-8)
        return loss

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, device, learning_rate=1e-3, 
                 max_iterations=250000, checkpoint_dir='checkpoints', checkpoint_every=5000):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.max_iterations = max_iterations
        self.note_loss_fn = FocalLoss(alpha=1.0, gamma=2.0)
        self.velocity_loss_fn = VelocityLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, fused=True)  # NEW: fused=True for faster Adam
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_val_loss = float('inf')
        self.iteration = 0
        self.train_losses = []
        self.val_losses = []
        self.checkpoint_every = checkpoint_every
        self.interrupted = False
        
        # NEW: Mixed precision scaler
        self.scaler = GradScaler()
        
        # NEW: Signal handling for graceful exit
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
        
        # NEW: Torch compile for PyTorch 2.0+
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
    def _handle_interrupt(self, signum, frame):
        print("\nGraceful shutdown initiated...")
        self.interrupted = True
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        epoch_losses = []  # NEW: Store per-batch losses for better logging
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for mel_spec, state_labels, context in pbar:
            if self.interrupted:
                print("Training interrupted mid-epoch. Saving progress...")
                break
                
            mel_spec = mel_spec.to(self.device, non_blocking=True)
            state_labels = state_labels.to(self.device, non_blocking=True)
            context = context.to(self.device, non_blocking=True)
            
            # NEW: Mixed precision forward pass
            with autocast():
                state_probs, velocities = self.model(mel_spec, context)
                loss = self.note_loss_fn(state_probs, state_labels)
            
            # NEW: Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)  # NEW: More efficient than zero_grad()
            
            loss_item = loss.item()
            total_loss += loss_item
            num_batches += 1
            self.iteration += 1
            
            epoch_losses.append(loss_item)
            pbar.set_postfix({'loss': f'{loss_item:.4f}', 'iter': self.iteration})
            
            # NEW: Save latest checkpoint periodically (not just at epoch end)
            if self.iteration % self.checkpoint_every == 0:
                self.save_checkpoint('latest_model.pt', include_losses=True)
            
            if self.iteration >= self.max_iterations:
                break
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            # NEW: Use autocast for validation too
            for mel_spec, state_labels, context in tqdm(self.val_loader, desc='Validation', leave=False):
                mel_spec = mel_spec.to(self.device, non_blocking=True)
                state_labels = state_labels.to(self.device, non_blocking=True)
                context = context.to(self.device, non_blocking=True)
                
                with autocast():
                    state_probs, velocities = self.model(mel_spec, context)
                    loss = self.note_loss_fn(state_probs, state_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def test(self):
        """NEW: Final test evaluation"""
        print("\nRunning final test evaluation...")
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for mel_spec, state_labels, context in tqdm(self.test_loader, desc='Testing'):
                mel_spec = mel_spec.to(self.device, non_blocking=True)
                state_labels = state_labels.to(self.device, non_blocking=True)
                context = context.to(self.device, non_blocking=True)
                
                with autocast():
                    state_probs, velocities = self.model(mel_spec, context)
                    loss = self.note_loss_fn(state_probs, state_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Final Test Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, filename, include_losses=False):
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'scaler_state_dict': self.scaler.state_dict(),  # NEW: Save scaler state
        }
        
        if include_losses:
            checkpoint['train_losses'] = self.train_losses
            checkpoint['val_losses'] = self.val_losses
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, filename):
        """NEW: Load checkpoint to resume training"""
        path = self.checkpoint_dir / filename
        if not path.exists():
            print(f"No checkpoint found at {path}, starting from scratch...")
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']
        
        print(f"\nResumed from iteration {self.iteration:,} with best val loss {self.best_val_loss:.4f}")
        return True

    def save_losses(self, filename='losses.npy'):
        np.save(self.checkpoint_dir / filename, {
            'train_losses': np.array(self.train_losses),
            'val_losses': np.array(self.val_losses)
        })
        print(f"Losses saved: {self.checkpoint_dir / filename}")

    def train(self, resume=False):
        print(f"\nStarting training for {self.max_iterations:,} iterations...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        
        # Try to resume if requested
        if resume:
            self.load_checkpoint('latest_model.pt')
        
        epoch = 0
        start_epoch = self.iteration // len(self.train_loader)
        
        try:
            while self.iteration < self.max_iterations and not self.interrupted:
                epoch += 1
                actual_epoch = start_epoch + epoch
                print(f"\n=== Epoch {actual_epoch} (Iteration {self.iteration:,}) ===")
                
                # TRAIN FIRST
                avg_train_loss = self.train_epoch()
                print(f"Average training loss: {avg_train_loss:.4f}")
                
                # THEN VALIDATE
                val_loss = self.validate()
                print(f"Validation Loss: {val_loss:.4f} (Best: {self.best_val_loss:.4f})")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print("New best model saved!")
                
                # Save epoch checkpoint
                if epoch % 5 == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{actual_epoch}.pt')
                
                # Always save latest state
                self.save_checkpoint('latest_model.pt', include_losses=True)
                
                # Early stopping check (optional)
                if len(self.val_losses) > 10 and val_loss > min(self.val_losses[-10:]):
                    print("No improvement in 10 epochs, consider early stopping")
        
        except Exception as e:
            print(f"\nTraining crashed with error: {e}")
            self.save_checkpoint('emergency_checkpoint.pt', include_losses=True)
            raise
        
        finally:
            # Save final state regardless of how training ended
            self.save_checkpoint('latest_model.pt', include_losses=True)
            
            if self.interrupted:
                print("\nTraining was interrupted by user.")
                print("Run with resume=True to continue from the last checkpoint.")
            else:
                print("\nTraining completed successfully!")
                # FINAL TEST EVALUATION
                self.test()
            
            # Save final losses
            self.save_losses()

# --- Main Function (updated) ---
def main(resume=False):
    config = {
        'data_dir': 'dataset/preprocessed_dataset.h5',
        'batch_size': 12,  # DRAMATICALLY INCREASED from 4
        'learning_rate': 1e-3,
        'max_iterations': 250000,
        'num_workers': 8,  # INCREASED for better data loading
        'sequence_length': 312,
        'checkpoint_dir': 'checkpoints',
        'checkpoint_every': 5000,  # NEW: Save every 5000 iterations
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # NEW: Print GPU info
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Enable benchmark mode for better performance
    torch.backends.cudnn.benchmark = True

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        sequence_length=config['sequence_length']
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    model = PARModel()
    
    # NEW: Move model to device before counting parameters
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,  # NEW: Pass test_loader
        device=device,
        learning_rate=config['learning_rate'],
        max_iterations=config['max_iterations'],
        checkpoint_dir=config['checkpoint_dir'],
        checkpoint_every=config['checkpoint_every'],
    )
    trainer.train(resume=resume)

if __name__ == "__main__":
    # NEW: Parse resume from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()
    
    main(resume=args.resume)