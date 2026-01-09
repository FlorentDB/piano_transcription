import pyaudio
import wave
import numpy as np
from typing import Optional
import sys
import tty
import termios
import threading

class AudioProcessor:
    """Real-time audio capture and processing with keyboard controls."""
    
    def __init__(
        self,
        sample_rate: int = 44100,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        self.audio = pyaudio.PyAudio()
        self.frames = []
        self.stream = None
        self.is_recording = False
        self.is_paused = False
        self.should_quit = False
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Override this method to implement custom signal processing."""
        return data
    
    def _callback(self, in_data, frame_count, time_info, status):
        """Internal callback for audio stream."""
        if self.is_recording and not self.is_paused:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            processed = self.process(audio_data)
            self.frames.append(in_data)
            return (processed.tobytes(), pyaudio.paContinue)
        return (in_data, pyaudio.paContinue)
    
    def start_recording(self) -> None:
        """Start recording audio."""
        if not self.is_recording:
            self.frames = []
            self.is_recording = True
            self.is_paused = False
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._callback
            )
            self.stream.start_stream()
            print("Recording started...")
    
    def pause_recording(self) -> None:
        """Pause/resume recording."""
        if self.is_recording:
            self.is_paused = not self.is_paused
            if self.is_paused:
                print("⏸Recording paused")
            else:
                print("Recording resumed")
    
    def stop_recording(self) -> None:
        """Stop recording."""
        if self.is_recording:
            self.is_recording = False
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            print("Recording stopped")
    
    def get_devices(self) -> list:
        """List all available input devices."""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0: #type:ignore
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels']
                })
        return devices
    
    def close(self) -> None:
        """Release audio resources."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_key():
    """Get a single keypress from stdin (Linux/Unix)."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    """Main function with keyboard controls."""
    
    # List available devices
    temp_processor = AudioProcessor()
    print("Available input devices:")
    for device in temp_processor.get_devices():
        print(f"  [{device['index']}] {device['name']}")
    print()
    temp_processor.close()
    
    # Select device
    device_to_use = 11
    
    processor = AudioProcessor(device_index=device_to_use)
    
    print("=" * 50)
    print("AUDIO RECORDER - KEYBOARD CONTROLS")
    print("=" * 50)
    print("Press 'r' to start recording")
    print("Press 'SPACE' to pause/resume recording")
    print("Press 'q' to quit")
    print("=" * 50)
    print()
    
    try:
        while not processor.should_quit:
            key = get_key()
            
            if key == 'r':
                processor.start_recording()
            elif key == ' ':  # Space key
                processor.pause_recording()
            elif key == 'q':
                processor.should_quit = True
                break
            elif key == '\x03':  # Ctrl+C
                processor.should_quit = True
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        processor.stop_recording()
        processor.close()
        print("\nRecording Ended")


if __name__ == "__main__":
    main()