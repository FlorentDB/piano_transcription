"""
pic_detection.py

Détection de pics à partir d'une CQT calculée depuis un spectrogramme.
"""

import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage as ndi


import soundfile as sf
import librosa


def load_wav(path, sr = None, mono = True):
    """Charge un fichier WAV depuis le disque.

    Parameters
    ----------
    path : str
        Chemin vers le fichier .wav
    sr : int | None
        Taux d'échantillonnage cible. Si None, on conserve le sr du fichier.
    mono : bool
        Si True, convertit en mono en moyennant les canaux.

    Returns
    -------
    y : np.ndarray
        Signal audio (float32, -1..1)
    sr : int
        Taux d'échantillonnage du signal retourné
    """
    y, sr_orig = sf.read(path, dtype='float32')

    # si stéréo et mono demandé
    if y.ndim > 1 and mono:
        y = np.mean(y, axis=1)

    # resampling si demandé
    if sr is not None and sr != sr_orig:
        y = librosa.resample(y, orig_sr=sr_orig, target_sr=sr)
        sr_orig = sr

    return y, sr_orig


def compute_cqt_librosa(y,
                        sr,
                        n_bins = 84,
                        bins_per_octave = 12,
                        hop_length = 512,
                        fmin  = None,
                        filter_scale = 1,
                        norm  = 1,
                        cqt=True,
                        gamma=None):
    """Calcule la Constant-Q Transform (CQT) avec librosa.

    Parameters
    ----------
    y : np.ndarray
        Signal audio (mono)
    sr : int
        Taux d'échantillonnage du signal
    n_bins : int
        Nombre total de bins (résolution en fréquence)
    bins_per_octave : int
        Nombre de bins par octave (affecte la résolution)
    hop_length : int
        Pas (frames) entre deux trames CQT
    fmin : float | 32
        Fréquence minimale. Si None, prend "C1" (~32.7 Hz)
    filter_scale : int
        Échelle des filtres (paramètre librosa)
    norm : int | None
        Normalisation des filtres (paramètre librosa)

    Returns
    -------
    C : np.ndarray
        Matrice CQT complexe de forme (n_bins, n_frames)
    freqs : np.ndarray
        Fréquences centrales de chaque bin (Hz)
    times : np.ndarray
        Temps correspondant à chaque frame (s)
    """

    if fmin is None:
        fmin = 32.7

    if cqt:
        # Calcul CQT
        C = librosa.cqt(y=y,
                        sr=sr,
                        hop_length=hop_length,
                        fmin=fmin,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        filter_scale=filter_scale,
                        norm=norm)
    else:
        # Calcul VQT
        if gamma is None:
            gamma = 20  # Valeur par défaut optimale pour piano
        C = librosa.vqt(y=y,
                        sr=sr,
                        hop_length=hop_length,
                        fmin=fmin,
                        n_bins=n_bins,
                        bins_per_octave=bins_per_octave,
                        gamma=gamma,
                        norm=norm)

    freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop_length)

    return C, freqs, times


def cqt_to_db(C: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Convertit une CQT complexe en échelle décibels (magnitude).

    Parameters
    ----------
    C : np.ndarray
        Matrice CQT complexe
    ref : float
        Référence pour la conversion en dB (par défaut 1.0)
    amin : float
        Valeur minimale d'amplitude pour éviter log(0)
    top_db : float
        Plage dynamique maximale (dB) pour tronquer

    Returns
    -------
    S_db : np.ndarray
        Spectrogramme CQT en dB (shape=(n_bins, n_frames))
    """
    magnitude = np.abs(C)
    S_db = librosa.amplitude_to_db(magnitude, ref=ref, amin=amin, top_db=top_db)
    return S_db


# Fonction de commodité : depuis un fichier WAV -> CQT (complexe ou dB)
def compute_cqt_from_wav(path,
                         sr = None,
                         n_bins = 84,
                         bins_per_octave  = 12,
                         hop_length  = 512,
                         fmin  = None,
                         return_db = False,
                         **cqt_kwargs):
    """Charge un .wav et calcule sa CQT en une seule étape.

    Parameters
    ----------
    path : str
        Chemin vers le .wav
    sr : int | None
        Taux d'échantillonnage cible
    return_db : bool
        Si True, renvoie la CQT en dB (magnitude), sinon la CQT complexe
    cqt_kwargs : dict
        Paramètres supplémentaires passés à `compute_cqt_librosa`

    Returns
    -------
    S : np.ndarray
        CQT (complexe) ou CQT en dB (float)
    freqs : np.ndarray
        Fréquences par bin (Hz)
    times : np.ndarray
        Temps par frame (s)
    """
    y, sr_used = load_wav(path, sr=sr, mono=True)
    C, freqs, times = compute_cqt_librosa(y=y,
                                         sr=sr_used,
                                         n_bins=n_bins,
                                         bins_per_octave=bins_per_octave,
                                         hop_length=hop_length,
                                         fmin=fmin,
                                         **cqt_kwargs)
    if return_db:
        return cqt_to_db(C), freqs, times
    return C, freqs, times

