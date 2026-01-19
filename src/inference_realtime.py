import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from model.model import PARModel
from audio_processor import AudioProcessor, get_key
import argparse
import time
import os
from collections import deque

class RealtimeTranscription:
    def __init__(self, model_path, device='cuda', sample_rate=16000, 
                 hop_length=512, n_mels=700, chunk_duration=0.05):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.chunk_samples = int(sample_rate * chunk_duration)
        
        # CRITICAL: These must match your training parameters exactly
        self.n_fft = 2048
        self.f_min = 27.5  # A0
        self.f_max = sample_rate // 2
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = PARModel()
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove '_orig_mod.' prefix if present (from torch.compile)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            state_dict = {
                key.replace('_orig_mod.', ''): value 
                for key, value in state_dict.items()
            }
            print("Removed '_orig_mod.' prefix from compiled model checkpoint")
        
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(device)
        
        # Optional: Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
        # Mel spectrogram transform (must match training preprocessing)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0  # Power spectrogram
        ).to(device)
        
        # Audio buffer (ring buffer to hold recent audio)
        buffer_duration = 2.0  # Keep last 2 seconds
        self.audio_buffer = deque(maxlen=int(sample_rate * buffer_duration))
        
        # Timing and state
        self.reset_context()
        self.last_process_time = 0
        self.process_interval = 0.05  # Process every 50ms
        
    def reset_context(self):
        """Reset autoregressive context (clears active notes)"""
        self.prev_context = torch.zeros(1, 88, 3, device=self.device)
        self.active_notes = {}
        self.session_start_time = time.time()
        print("\n Context reset] All notes cleared, starting fresh...")
        
    def process_audio(self, audio_int16: np.ndarray):
        """Process incoming audio chunk from PyAudio callback"""
        # Normalize int16 to float32 [-1, 1]
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        # Add to buffer
        self.audio_buffer.extend(audio_float)
        
        # Rate-limit processing
        current_time = time.time()
        if current_time - self.last_process_time < self.process_interval:
            return
            
        # Process if we have enough samples
        if len(self.audio_buffer) >= self.n_fft:
            self.last_process_time = current_time
            self._transcribe_frame()
            
    def _transcribe_frame(self):
        """Internal: Run model on recent audio"""
        try:
            # Get recent audio segment
            # For mel spectrogram, we need at least n_fft samples
            audio_tensor = torch.tensor(
                list(self.audio_buffer)[-self.n_fft:], 
                device=self.device, 
                dtype=torch.float32
            ).unsqueeze(0)  # (1, n_fft)
            
            # Compute log mel spectrogram
            with torch.no_grad(), torch.cuda.amp.autocast():
                mel_spec = self.mel_transform(audio_tensor)  # (1, n_mels, T)
                mel_spec = torch.clamp(mel_spec, min=1e-5)
                log_mel = torch.log(mel_spec)  # Use log, not dB
                
                # Model expects (B, C, F, T) where C=1
                mel_input = log_mel.unsqueeze(0)  # (1, 1, n_mels, T)
                
                # Run inference
                state_probs, velocity_pred = self.model(
                    mel_input, 
                    self.prev_context,
                    teacher_forcing=False
                )
                
                # Use last time step
                current_states = torch.argmax(state_probs[..., -1], dim=-1)  # (1, 88)
                current_velocities = velocity_pred[..., -1].squeeze(-1)  # (1, 88)
                
                # Update context for next frame
                self._update_context(current_states, current_velocities)
                
                # Decode to MIDI events
                self._decode_events(current_states, current_velocities)
                
        except Exception as e:
            print(f"Error in transcription: {e}")
            
    def _update_context(self, states: torch.Tensor, velocities: torch.Tensor):
        """Update autoregressive context based on predictions"""
        # Normalize state to [0,1]
        context_states = states.float() / 4.0
        
        # Duration logic: increment for active notes, reset for inactive
        active_mask = ((states > 0) & (states < 4)).float()
        time_delta = self.hop_length / self.sample_rate
        
        # Update duration (clipped at model.max_duration_seconds)
        self.prev_context[:, :, 1] += active_mask * (time_delta / self.model.max_duration_seconds)
        self.prev_context[:, :, 1] = torch.clamp(self.prev_context[:, :, 1], max=1.0)
        
        # Reset duration for off/offset notes
        reset_mask = (states == 0) | (states == 4)
        self.prev_context[:, :, 1][reset_mask] = 0
        
        # Update velocity and state
        self.prev_context[:, :, 2] = velocities
        self.prev_context[:, :, 0] = context_states
        
    def _decode_events(self, states: torch.Tensor, velocities: torch.Tensor):
        """Decode state predictions to note on/off events"""
        states_np = states.cpu().numpy()[0]  # (88,)
        velocities_np = velocities.cpu().numpy()[0]  # (88,)
        
        elapsed = time.time() - self.session_start_time
        
        for pitch_idx in range(88):
            state = int(states_np[pitch_idx])
            velocity = velocities_np[pitch_idx] * 127.0
            
            midi_note = pitch_idx + 21  # 21=A0, 108=C8
            
            # Note ON: onset(1) or re-onset(2)
            if state in {1, 2} and midi_note not in self.active_notes:
                self.active_notes[midi_note] = {
                    'velocity': int(velocity),
                    'time': elapsed
                }
                note_name = self._midi_to_note(midi_note)
                print(f" NOTE ON  {midi_note:3d} ({note_name:>3s}) | Vel: {int(velocity):3d} | t={elapsed:.2f}s")
            
            # Note OFF: offset(4) or off(0)
            elif state in {0, 4} and midi_note in self.active_notes:
                note_info = self.active_notes.pop(midi_note)
                duration = elapsed - note_info['time']
                note_name = self._midi_to_note(midi_note)
                print(f"  NOTE OFF {midi_note:3d} ({note_name:>3s}) | Duration: {duration:.2f}s")
                
    def _midi_to_note(self, midi_num):
        """Convert MIDI number to note name (60 -> C4)"""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_num // 12) - 1
        note = notes[midi_num % 12]
        return f"{note}{octave}"

class TranscriptionAudioProcessor(AudioProcessor):
    """AudioProcessor that sends audio to transcription model"""
    
    def __init__(self, transcription_model, **kwargs):
        super().__init__(**kwargs)
        self.transcription_model = transcription_model
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Called by PyAudio callback for each audio chunk"""
        # Send audio to transcription model (non-blocking)
        self.transcription_model.process_audio(data)
        
        # Return unmodified audio (or process if needed)
        return data

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Piano Transcription",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        choices=['cuda', 'cpu'], help='Inference device (auto-detected)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (must match training)')
    parser.add_argument('--list_devices', action='store_true',
                        help='List audio devices and exit')
    parser.add_argument('--device_index', type=int, default=None,
                        help='Audio input device index')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        temp = AudioProcessor()
        print("\nAvailable audio input devices:")
        for dev in temp.get_devices():
            print(f"  [{dev['index']:2d}] {dev['name']}")
        temp.close()
        return
    
    # Initialize transcription
    print("\n=== Initializing Real-time Piano Transcription ===")
    transcription = RealtimeTranscription(
        model_path=args.model_path,
        device=args.device,
        sample_rate=args.sample_rate
    )
    
    # Get audio device
    if args.device_index is None:
        temp = AudioProcessor()
        devices = temp.get_devices()
        print("\nAvailable audio input devices:")
        for dev in devices:
            print(f"  [{dev['index']:2d}] {dev['name']}")
        temp.close()
        
        idx = input("\nEnter device index: ").strip()
        args.device_index = int(idx)
    
    # Create audio processor
    processor = TranscriptionAudioProcessor(
        transcription_model=transcription,
        sample_rate=args.sample_rate,
        device_index=args.device_index,
        chunk_size=1024
    )
    
    # Control interface
    print("\n" + "="*60)
    print(" REAL-TIME PIANO TRANSCRIPTION")
    print("="*60)
    print("Press 'r' to start/stop transcription")
    print("Press 'c' to reset context (clear active notes)")
    print("Press 'q' to quit")
    print("="*60)
    print("\nReady. Press 'r' to begin...")
    
    try:
        while True:
            key = get_key()
            
            if key == 'r':
                if not processor.is_recording:
                    transcription.reset_context()
                    processor.start_recording()
                    print("\n TRANSCRIPTION ACTIVE - Play piano...")
                else:
                    processor.stop_recording()
                    print("\n  Transcription stopped")
                    
            elif key == 'c':
                transcription.reset_context()
                
            elif key in ('q', '\x03'):  # 'q' or Ctrl+C
                print("\nShutting down...")
                break
                
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        processor.stop_recording()
        processor.close()
        print("\n Audio resources released. Goodbye!")

if __name__ == "__main__":
    main()