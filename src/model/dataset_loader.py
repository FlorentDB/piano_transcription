import os
import random
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# DEFAULTS
H5_DEFAULT = "dataset/preprocessed_dataset.h5"
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RNG_SEED = 42
PAD_DB = -80.0  # padding value for log-mel (dB)
MAX_DURATION_FRAMES = 5.0 * (16000 / 512) # 5 seconds in frames

def _list_all_chunk_keys(h5_path: str) -> List[str]:
    keys = []
    with h5py.File(h5_path, "r") as f:
        for dataset_name in f.keys():  # e.g., 'MAESTRO', 'SMD'
            root = f[dataset_name]
            for piece in root.keys():
                piece_group = root[piece]
                for chunk in piece_group.keys():
                    chunk_key = f"{dataset_name}/{piece}/{chunk}"
                    # validate existence
                    if 'spectrogram' in piece_group[chunk].keys() and 'multi_state' in piece_group[chunk].keys():
                        keys.append(chunk_key)
    return keys

def _split_keys_random(keys, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO, seed=RNG_SEED):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    rng = random.Random(seed)
    keys = keys.copy()
    rng.shuffle(keys)
    n = len(keys)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = keys[:n_train]
    val = keys[n_train:n_train + n_val]
    test = keys[n_train + n_val:]
    return train, val, test

def compute_context_from_states(state_labels, max_duration=MAX_DURATION_FRAMES):
    """
    Compute recursive context [state, duration, velocity] for each frame.
    
    Args:
        state_labels: (88, T) - state labels (0=off, 1=onset, 2=reonset, 3=sustain, 4=offset)
        max_duration: maximum duration to track in frames
    
    Returns:
        context: (88, 3, T) - [state, duration_norm, velocity_norm] for each frame
    """
    num_pitches, T = state_labels.shape
    context = torch.zeros((num_pitches, 3, T), dtype=torch.float32)
    
    # Track current state for each pitch
    current_state = torch.zeros(num_pitches, dtype=torch.long)  # 0 = off
    current_duration = torch.zeros(num_pitches, dtype=torch.float32)
    current_velocity = torch.zeros(num_pitches, dtype=torch.float32)
    
    for t in range(T):
        # Store current context
        context[:, 0, t] = current_state.float()
        context[:, 1, t] = torch.clamp(current_duration / max_duration, 0, 1)
        context[:, 2, t] = current_velocity
        
        # Update state for next frame
        for pitch in range(num_pitches):
            state = state_labels[pitch, t].item()
            
            if state == 1 or state == 2:  # onset or reonset
                current_state[pitch] = state
                current_duration[pitch] = 1.0
                # Velocity would come from onset metadata - for now use default
                current_velocity[pitch] = 0.5
            elif state == 3:  # sustain
                current_state[pitch] = state
                current_duration[pitch] += 1.0
            elif state == 4:  # offset
                current_state[pitch] = state
                current_duration[pitch] = 0.0
                current_velocity[pitch] = 0.0
            else:  # off
                current_state[pitch] = 0
                current_duration[pitch] = 0.0
                current_velocity[pitch] = 0.0
    
    return context


class H5ChunkDataset(Dataset):
    def __init__(self, h5_path: str, keys: List[str], sequence_length: int = 312, split: str = "train"):
        super().__init__()
        assert split in ("train", "val", "test")
        self.h5_path = h5_path
        self.keys = keys
        self.sequence_length = int(sequence_length)
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
            if self.split == "train":
                start = random.randint(0, T_chunk - seq_len)
            else:
                start = (T_chunk - seq_len) // 2
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

        # FIXED: Compute proper context from state labels
        context = compute_context_from_states(state_t)  # (88, 3, T)
        
        # Extract initial context (at t=0) for autoregressive model
        initial_context = context[:, :, 0]  # (88, 3)

        return spec_t, state_t, initial_context

    def __del__(self):
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass

def create_dataloaders(
    data_dir: str,
    batch_size: int = 12,
    num_workers: int = 4,
    sequence_length: int = 312,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RNG_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    h5_path = data_dir if data_dir is not None else H5_DEFAULT
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    keys = _list_all_chunk_keys(h5_path)
    if len(keys) == 0:
        raise RuntimeError(f"No chunks found in {h5_path}")

    train_keys, val_keys, test_keys = _split_keys_random(keys, train_ratio, val_ratio, test_ratio, seed)

    train_ds = H5ChunkDataset(h5_path, train_keys, sequence_length=sequence_length, split='train')
    val_ds = H5ChunkDataset(h5_path, val_keys, sequence_length=sequence_length, split='val')
    test_ds = H5ChunkDataset(h5_path, test_keys, sequence_length=sequence_length, split='test')

    def _worker_init_fn(worker_id):
        seed_worker = seed + worker_id
        random.seed(seed_worker)
        np.random.seed(seed_worker)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True,
                            worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True,
                             worker_init_fn=_worker_init_fn)

    return train_loader, val_loader, test_loader
