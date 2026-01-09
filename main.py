from src.CQT_Pic.CQT import compute_cqt_from_wav
from src.CQT_Pic.pic_detection import plot_piano_roll, notes_to_midi, detect_notes
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

image_path = "assets/images"
audio_path = "assets/audio"
audio = os.path.join(audio_path,"Dance of the Sugar Plum Fairy Both Hands AUDIO.wav")
C, freqs, time = compute_cqt_from_wav(audio)

# Detect notes
notes = detect_notes(C, freqs, time, threshold_db=-40, min_duration=0.05)

# Create a figure
plt.figure(figsize=(10, 6))

# Display the spectrogram
librosa.display.specshow(librosa.amplitude_to_db(C, ref=np.max),
                         y_axis='cqt_note', x_axis='time')

# Save the figure
plt.savefig(os.path.join(image_path, 'cqt_spectrogram.png'), bbox_inches='tight', dpi=300)
plt.close()

# Create and save piano roll
plot_piano_roll(notes, save_path=os.path.join(image_path, 'piano_roll.png'))

# Convert and save to MIDI
notes_to_midi(notes, output_path=os.path.join(audio_path, 'transcription.mid'), tempo=100)