import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
from pathlib import Path
from typing import Tuple, List, Optional
import json

class PianoRollProcessor:
    """Convertit MIDI en piano roll avec les 5 états de notes"""
    
    def __init__(self, fps=31.25, pedal_threshold=64):
        """
        Args:
            fps: frames per second (16000 / 512 = 31.25)
            pedal_threshold: seuil pour la pédale sustain
        """
        self.fps = fps
        self.pedal_threshold = pedal_threshold
        self.num_pitches = 88
        self.min_pitch = 21  # A0
        
    def midi_to_states(self, midi_path: str, audio_length: float) -> np.ndarray:
        """
        Convertit un fichier MIDI en séquence d'états (5 classes)
        
        Returns:
            states: (88, num_frames) avec valeurs 0-4
                0: off
                1: onset
                2: sustain
                3: re-onset
                4: offset
        """
        midi = pretty_midi.PrettyMIDI(midi_path)
        num_frames = int(audio_length * self.fps)
        
        # Initialize avec état "off" (0)
        states = np.zeros((self.num_pitches, num_frames), dtype=np.int64)
        
        # Extraire les événements de pédale sustain
        pedal_intervals = self._extract_pedal_intervals(midi)
        
        # Traiter chaque note
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                pitch_idx = note.pitch - self.min_pitch
                if pitch_idx < 0 or pitch_idx >= self.num_pitches:
                    continue
                
                onset_frame = int(note.start * self.fps)
                offset_frame = int(note.end * self.fps)
                
                # Elonger l'offset si pédale sustain active
                offset_frame = self._apply_pedal(
                    offset_frame, pedal_intervals, note.end
                )
                
                # Limiter aux bornes
                onset_frame = max(0, min(onset_frame, num_frames - 1))
                offset_frame = max(0, min(offset_frame, num_frames - 1))
                
                # Vérifier si c'est un re-onset
                is_reonset = self._check_reonset(
                    states[pitch_idx], onset_frame
                )
                
                # Marquer onset
                if is_reonset:
                    states[pitch_idx, onset_frame] = 3  # re-onset
                else:
                    states[pitch_idx, onset_frame] = 1  # onset
                
                # Marquer sustain
                if onset_frame + 1 < offset_frame:
                    states[pitch_idx, onset_frame+1:offset_frame] = 2  # sustain
                
                # Marquer offset
                if offset_frame < num_frames:
                    states[pitch_idx, offset_frame] = 4  # offset
        
        return states
    
    def _extract_pedal_intervals(self, midi) -> List[Tuple[float, float]]:
        """Extrait les intervalles où la pédale sustain est active"""
        intervals = []
        
        for instrument in midi.instruments:
            pedal_changes = [
                (cc.time, cc.value) 
                for cc in instrument.control_changes 
                if cc.number == 64  # Sustain pedal
            ]
            
            if not pedal_changes:
                continue
            
            pedal_down = None
            for time, value in pedal_changes:
                if value >= self.pedal_threshold and pedal_down is None:
                    pedal_down = time
                elif value < self.pedal_threshold and pedal_down is not None:
                    intervals.append((pedal_down, time))
                    pedal_down = None
            
            # Si pédale reste enfoncée jusqu'à la fin
            if pedal_down is not None:
                intervals.append((pedal_down, midi.get_end_time()))
        
        return intervals
    
    def _apply_pedal(self, offset_frame: int, pedal_intervals: List, 
                     note_end_time: float) -> int:
        """Elonge l'offset si la pédale est active"""
        for pedal_start, pedal_end in pedal_intervals:
            if pedal_start <= note_end_time <= pedal_end:
                return int(pedal_end * self.fps)
        return offset_frame
    
    def _check_reonset(self, pitch_states: np.ndarray, onset_frame: int) -> bool:
        """Vérifie si c'est un re-onset (note déjà active)"""
        if onset_frame == 0:
            return False
        # Re-onset si la note était en sustain juste avant
        return pitch_states[onset_frame - 1] == 2  # sustain


class MelSpectrogramExtractor:
    """Extrait le mel spectrogramme selon les specs de l'article"""
    
    def __init__(
        self,
        sample_rate=16000,
        n_fft=4096,
        hop_length=512,
        n_mels=700,
        fmin=27.5,
        fmax=8372.0
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        
    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extrait le log mel spectrogram
        
        Returns:
            mel_spec: (n_mels, num_frames)
        """
        # Charger l'audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Calculer mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        
        # Convertir en log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel_spec


class PianoTranscriptionDataset(Dataset):
    """Dataset pour transcription de piano selon l'article PAR"""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        sequence_length: int = 100,
        dataset_name: str = 'maestro'
    ):
        """
        Args:
            data_dir: chemin vers le dossier des données
            split: 'train', 'validation', ou 'test'
            sequence_length: longueur des séquences (en frames)
            dataset_name: 'maestro', 'maps', ou 'smd'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.sequence_length = sequence_length
        self.dataset_name = dataset_name
        
        # Initialiser les extracteurs
        self.mel_extractor = MelSpectrogramExtractor()
        self.roll_processor = PianoRollProcessor()
        
        # Charger la liste des fichiers
        self.file_list = self._load_file_list()
        
        print(f"Loaded {len(self.file_list)} files for {split} split")
    
    def _load_file_list(self) -> List[dict]:
        """Charge la liste des paires audio/MIDI selon le dataset"""
        
        if self.dataset_name == 'maestro':
            return self._load_maestro()
        elif self.dataset_name == 'maps':
            return self._load_maps()
        elif self.dataset_name == 'smd':
            return self._load_smd()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_maestro(self) -> List[dict]:
        """Charge MAESTRO dataset"""
        # Charger le fichier JSON de métadonnées
        json_path = self.data_dir / 'maestro-v3.0.0.json'
        
        if not json_path.exists():
            raise FileNotFoundError(f"MAESTRO metadata not found: {json_path}")
        
        with open(json_path, 'r') as f:
            metadata = json.load(f)
        
        file_list = []
        for entry in metadata:
            if entry['split'] == self.split:
                file_list.append({
                    'audio': self.data_dir / entry['audio_filename'],
                    'midi': self.data_dir / entry['midi_filename'],
                    'duration': entry['duration']
                })
        
        return file_list
    
    def _load_maps(self) -> List[dict]:
        """Charge MAPS dataset (acoustic piano uniquement)"""
        file_list = []
        
        # MAPS a une structure: MAPS/AkPnBcht/MUS/
        maps_root = self.data_dir / 'MAPS'
        acoustic_pianos = ['AkPnBcht', 'AkPnBsdf']  # Les 2 pianos acoustiques
        
        for piano in acoustic_pianos:
            mus_dir = maps_root / piano / 'MUS'
            if not mus_dir.exists():
                continue
            
            for wav_file in mus_dir.glob('*.wav'):
                midi_file = wav_file.with_suffix('.mid')
                if midi_file.exists():
                    # Calculer la durée
                    audio, sr = librosa.load(str(wav_file), sr=None, duration=1)
                    duration = librosa.get_duration(path=str(wav_file))
                    
                    file_list.append({
                        'audio': wav_file,
                        'midi': midi_file,
                        'duration': duration
                    })
        
        # Diviser en train/val/test (80/10/10)
        np.random.seed(42)
        indices = np.random.permutation(len(file_list))
        
        n_train = int(0.8 * len(file_list))
        n_val = int(0.1 * len(file_list))
        
        if self.split == 'train':
            file_list = [file_list[i] for i in indices[:n_train]]
        elif self.split == 'validation':
            file_list = [file_list[i] for i in indices[n_train:n_train+n_val]]
        else:  # test
            file_list = [file_list[i] for i in indices[n_train+n_val:]]
        
        return file_list
    
    def _load_smd(self) -> List[dict]:
        """Charge Saarland Music Dataset"""
        file_list = []
        
        smd_root = self.data_dir / 'SMD' / 'SMD_MIDI'
        
        for wav_file in smd_root.glob('**/*.wav'):
            # SMD a les MIDI dans le même dossier
            midi_file = wav_file.with_suffix('.mid')
            
            if midi_file.exists():
                duration = librosa.get_duration(path=str(wav_file))
                
                file_list.append({
                    'audio': wav_file,
                    'midi': midi_file,
                    'duration': duration
                })
        
        # Diviser en splits
        np.random.seed(42)
        indices = np.random.permutation(len(file_list))
        
        n_train = int(0.8 * len(file_list))
        n_val = int(0.1 * len(file_list))
        
        if self.split == 'train':
            file_list = [file_list[i] for i in indices[:n_train]]
        elif self.split == 'validation':
            file_list = [file_list[i] for i in indices[n_train:n_train+n_val]]
        else:
            file_list = [file_list[i] for i in indices[n_train+n_val:]]
        
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel_spec: (1, 700, T) - mel spectrogram
            state_labels: (88, T) - note states
            context: (88, 4) - [last_state, duration, velocity, unused]
        """
        file_info = self.file_list[idx]
        
        # Extraire mel spectrogram
        mel_spec = self.mel_extractor.extract(str(file_info['audio']))
        
        # Extraire piano roll avec états
        states = self.roll_processor.midi_to_states(
            str(file_info['midi']),
            file_info['duration']
        )
        
        # S'assurer que les tailles correspondent
        min_frames = min(mel_spec.shape[1], states.shape[1])
        mel_spec = mel_spec[:, :min_frames]
        states = states[:, :min_frames]
        
        # Extraire un segment aléatoire pendant l'entraînement
        if self.split == 'train' and min_frames > self.sequence_length:
            start_frame = np.random.randint(0, min_frames - self.sequence_length)
            mel_spec = mel_spec[:, start_frame:start_frame + self.sequence_length]
            states = states[:, start_frame:start_frame + self.sequence_length]
        
        # Créer le contexte initial (sera mis à jour pendant l'entraînement)
        context = self._create_initial_context(states)
        
        # Convertir en tenseurs
        mel_spec = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, 700, T)
        states = torch.LongTensor(states)  # (88, T)
        context = torch.FloatTensor(context)  # (88, 4)
        
        return mel_spec, states, context
    
    def _create_initial_context(self, states: np.ndarray) -> np.ndarray:
        """Crée le contexte initial [state, duration, velocity, unused]"""
        context = np.zeros((88, 4), dtype=np.float32)
        
        # Pour l'instant, contexte vide (sera mis à jour de manière autoregressive)
        # Dans l'entraînement, vous utiliserez teacher forcing
        
        return context


def create_dataloaders(
    data_dir: str,
    batch_size: int = 12,
    num_workers: int = 4,
    dataset_name: str = 'maestro',
    sequence_length: int = 100
):
    """
    Crée les dataloaders pour entraînement, validation et test
    
    Args:
        data_dir: chemin vers les données
        batch_size: taille du batch
        num_workers: nombre de workers pour le chargement
        dataset_name: 'maestro', 'maps', ou 'smd'
        sequence_length: longueur des séquences
    """
    
    train_dataset = PianoTranscriptionDataset(
        data_dir=data_dir,
        split='train',
        sequence_length=sequence_length,
        dataset_name=dataset_name
    )
    
    val_dataset = PianoTranscriptionDataset(
        data_dir=data_dir,
        split='validation',
        sequence_length=sequence_length,
        dataset_name=dataset_name
    )
    
    test_dataset = PianoTranscriptionDataset(
        data_dir=data_dir,
        split='test',
        sequence_length=sequence_length,
        dataset_name=dataset_name
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Exemple d'utilisation
if __name__ == "__main__":
    # Tester avec MAESTRO
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir='/path/to/maestro-v3.0.0',
        batch_size=4,
        num_workers=2,
        dataset_name='maestro'
    )
    
    # Vérifier un batch
    for mel_spec, states, context in train_loader:
        print(f"Mel spec shape: {mel_spec.shape}")  # (B, 1, 700, T)
        print(f"States shape: {states.shape}")  # (B, 88, T)
        print(f"Context shape: {context.shape}")  # (B, 88, 4)
        print(f"States unique values: {torch.unique(states)}")  # 0-4
        break
