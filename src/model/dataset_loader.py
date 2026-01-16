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

class H5ChunkDataset(Dataset):
    """
    Dataset that reads chunk-group entries from an HDF5 preprocessed file.

    Each item yields a tuple (mel_spec, state_labels, context) where:
      - mel_spec: FloatTensor (1, n_mels, T_seq)  (we keep channel dim first for conv2d)
      - state_labels: LongTensor (88, T_seq)
      - context: FloatTensor (88, 3)  (prev state, prev duration, prev velocity) - currently zeros

    Cropping:
      - If chunk_len >= sequence_length: random crop (train) or center crop (val/test).
      - If chunk_len < sequence_length: pad to sequence_length.
    """
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
            # open read-only
            self._h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        self._ensure_open()
        key = self.keys[idx]
        grp = self._h5[key]

        # load spectrogram: shape (T, n_mels)
        spec_np = np.array(grp['spectrogram'], dtype=np.float32)
        # load multi-state: (T, 88) or (frames, 88)
        multi_np = np.array(grp['multi_state'], dtype=np.uint8)

        T_chunk, n_mels = spec_np.shape
        # We want mel shape (1, n_mels, T_seq) eventually
        # We also want state_labels shape (88, T_seq)

        seq_len = self.sequence_length

        if T_chunk >= seq_len:
            # cropping
            if self.split == "train":
                start = random.randint(0, T_chunk - seq_len)
            else:
                start = (T_chunk - seq_len) // 2
            end = start + seq_len
            spec_slice = spec_np[start:end, :]   # (seq_len, n_mels)
            state_slice = multi_np[start:end, :] # (seq_len, 88)
        else:
            # will pad later
            spec_slice = spec_np
            state_slice = multi_np

        # convert to tensors and transpose/reshape
        # spec_slice: (seq_len_or_less, n_mels) -> (n_mels, seq_len_or_less) -> (1, n_mels, seq_len_or_less)
        spec_t = torch.from_numpy(spec_slice.T).float().unsqueeze(0)  # (1, n_mels, Tvar)
        # state_slice: (Tvar, 88) -> (88, Tvar)
        state_t = torch.from_numpy(state_slice.T).long()  # (88, Tvar)

        # padding if needed
        Tvar = spec_t.shape[-1]
        if Tvar < seq_len:
            pad_len = seq_len - Tvar
            # pad spectrogram on time axis (right pad)
            pad_spec = torch.full((1, n_mels, pad_len), PAD_DB, dtype=spec_t.dtype)
            spec_t = torch.cat([spec_t, pad_spec], dim=2)
            # pad states with OFF (0)
            pad_states = torch.zeros((88, pad_len), dtype=state_t.dtype)
            state_t = torch.cat([state_t, pad_states], dim=1)

        # create prev_context: (88,3) zeros [state, duration, velocity]
        context = torch.zeros((88, 3), dtype=torch.float32)

        # final shapes:
        # spec_t: (1, n_mels, seq_len)
        # state_t: (88, seq_len)
        # context: (88, 3)
        return spec_t, state_t, context

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
    dataset_name: str = "maestro",
    sequence_length: int = 312,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RNG_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test dataloaders from an HDF5 preprocessed file.

    Args:
      data_dir: path to HDF5 file (e.g. 'dataset/preprocessed_dataset.h5')
      batch_size, num_workers: DataLoader params
      dataset_name: unused here (kept for API compatibility)
      sequence_length: number of frames per training sample
    Returns:
      train_loader, val_loader, test_loader
    """
    h5_path = data_dir if data_dir is not None else H5_DEFAULT
    if not os.path.exists(h5_path):
        raise FileNotFoundError("f\HDF5 not found: {h5_path}")

    # list chunk keys
    keys = _list_all_chunk_keys(h5_path)
    if len(keys) == 0:
        raise RuntimeError("f\No chunks found in {h5_path}")

    # random chunk-level split
    train_keys, val_keys, test_keys = _split_keys_random(keys, train_ratio, val_ratio, test_ratio, seed)

    # create datasets
    train_ds = H5ChunkDataset(h5_path, train_keys, sequence_length=sequence_length, split='train')
    val_ds   = H5ChunkDataset(h5_path, val_keys, sequence_length=sequence_length, split='val')
    test_ds  = H5ChunkDataset(h5_path, test_keys, sequence_length=sequence_length, split='test')

    # worker init to set deterministic seeds per worker
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