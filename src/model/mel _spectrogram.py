# mel_spectrogram.py
"""
MelProcessor (librosa-based)
Consumes float32 audio chunks from a queue and produces mel frames.
Calls a user-provided callback `on_mel(mel_frame, timestamp)` for each produced mel frame.

Requires: numpy, librosa
"""

import numpy as np
import queue
import threading
import time
from typing import Callable, Optional

import librosa


class MelProcessor:
    def __init__(
        self,
        in_queue: queue.Queue,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 64,
        on_mel: Optional[Callable[[np.ndarray, float], None]] = None,
        use_db: bool = True,
        db_ref: float = 1.0,
        amin: float = 1e-10,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
    ):
        """
        Args:
            in_queue: Queue that yields float32 1-D numpy arrays (audio chunks).
            sample_rate: Sampling rate in Hz.
            n_fft: FFT window length.
            hop_length: Hop (step) in samples between consecutive mel frames.
            n_mels: Number of mel bands.
            on_mel: Callback: on_mel(mel_vector: np.ndarray, timestamp_s: float).
            use_db: If True convert mel-power to dB using librosa.power_to_db.
            db_ref: Reference value for power_to_db (default 1.0).
            amin: floor for power to avoid log(0).
            fmin/fmax: mel filter min/max frequency.
        """
        self.q = in_queue
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.on_mel = on_mel
        self.use_db = use_db
        self.db_ref = db_ref
        self.amin = amin
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(self.sr) / 2.0

        # Precompute Hann/Hamming window (user choice) and mel basis using librosa.
        self._window = np.hamming(self.n_fft)
        # librosa's mel filterbank: shape (n_mels, n_fft//2 + 1)
        self._mel_basis = librosa.filters.mel(
            sr=self.sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Running thread control
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._stop_event = threading.Event()
        self._started = False

        # small ring buffer for overlap handling
        self._buffer = np.zeros(0, dtype=np.float32)

    def start(self):
        """Start processing thread (returns immediately)."""
        if self._started:
            return
        self._stop_event.clear()
        self._thread.start()
        self._started = True
        print("MelProcessor: started")

    def stop(self):
        """Signal processing thread to stop and wait briefly."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("MelProcessor: stopped")

    def _compute_mel(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute mel-power (or mel-dB) for a single frame (length == n_fft).
        Returns a 1-D array of length n_mels.
        """
        # Apply window
        windowed = frame * self._window

        # Compute power spectrum (real FFT)
        S = np.abs(np.fft.rfft(windowed, n=self.n_fft)) ** 2  # power spectrum, shape (n_fft//2+1,)

        # Apply mel filterbank: result is mel-band energies (power)
        mel_power = np.dot(self._mel_basis, S)

        if self.use_db:
            # Convert to dB using librosa utility (10*log10 for power)
            mel_db = librosa.power_to_db(mel_power, ref=self.db_ref, amin=self.amin)
            return mel_db.astype(np.float32)
        else:
            return mel_power.astype(np.float32)

    def _loop(self):
        """
        Main loop: read chunks from queue, append to buffer, and while enough samples
        compute mel frames every hop_length samples. Estimates timestamp from samples consumed.
        """
        samples_consumed = 0  # for timestamp approximation
        while not self._stop_event.is_set():
            try:
                chunk = self.q.get(timeout=0.1)  # expects float32 mono chunk
            except queue.Empty:
                continue

            # Ensure correct dtype
            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32)

            # Append chunk to internal buffer
            self._buffer = np.concatenate((self._buffer, chunk))

            # While we have at least one full frame
            while len(self._buffer) >= self.n_fft:
                frame = self._buffer[: self.n_fft].copy()
                timestamp = samples_consumed / float(self.sr)  # approximate time of frame start in seconds

                # Compute mel vector
                try:
                    mel = self._compute_mel(frame)
                except Exception as ex:
                    # Protect loop from unexpected errors in computation
                    print("MelProcessor: error computing mel:", ex)
                    mel = None

                # Advance buffer by hop_length
                self._buffer = self._buffer[self.hop_length :]
                samples_consumed += self.hop_length

                # Call user callback (keep it non-blocking in user code)
                if mel is not None and self.on_mel is not None:
                    try:
                        self.on_mel(mel, timestamp)
                    except Exception as e:
                        # don't let user callback kill the processing loop
                        print("MelProcessor.on_mel exception:", e)

        # On exit, optionally process remaining partial frames (not done here)
