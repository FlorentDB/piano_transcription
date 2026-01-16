#!/usr/bin/env python3


import os
import json
import math
from glob import glob
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
import pretty_midi
import h5py
from tqdm import tqdm


# ----------------------------- Configuration ------------------------------
# Edit these paths directly before running
MAESTRO_DIR = '/home/spycologue/dataset/maestro-v3.0.0'   # <-- change me
SMD_DIR = '/home/spycologue/dataset/SMD'              # <-- change me
OUT_DIR = './dataset'                 # <-- change me (folder will be created)
CHUNK_SECONDS = 10.0

# ----------------------------- Parameters ---------------------------------
SR = 16000
N_FFT = 4096
HOP = 512
N_MELS = 700
FMIN = 27.5
FMAX = 8372

PIANO_MIN = 21
PIANO_MAX = 108
N_KEYS = PIANO_MAX - PIANO_MIN + 1  # 88

# Multi-state codes
OFF = 0
ONSET = 1
REONSET = 2
SUSTAIN = 3
OFFSET = 4


# ----------------------------- Helpers -----------------------------------

def compute_log_mel(audio, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=FMIN, fmax=FMAX):
    """Compute log-mel spectrogram (dB) with librosa. Returns (T, n_mels) float32.
    T corresponds to number of frames for hop_length.
    """
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                       power=2.0, n_mels=n_mels, fmin=fmin, fmax=fmax)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db.T.astype(np.float32)


def midi_collect_control_changes(pm: pretty_midi.PrettyMIDI, control_number=64):
    """Collect control change (sustain pedal) events across all instruments in the PrettyMIDI object.
    Returns list of (time, value) sorted by time. If none found, returns []
    """
    events = []
    for inst in pm.instruments:
        if hasattr(inst, 'control_changes'):
            for cc in inst.control_changes:
                if getattr(cc, 'number', None) == control_number:
                    events.append((cc.time, getattr(cc, 'value', 0)))
    events.sort(key=lambda x: x[0])
    return events


def build_sustain_boolean(events, total_seconds, frame_duration, threshold=64):
    """From (time,value) events return boolean array of length T_frames indicating sustain active.
    We consider value >= threshold -> pedal ON.
    """
    n_frames = int(math.ceil(total_seconds / frame_duration))
    sustain = np.zeros(n_frames, dtype=np.bool_)
    if len(events) == 0:
        return sustain
    on = False
    last_time = 0.0
    for t, v in events:
        start_frame = int(math.floor(last_time / frame_duration))
        end_frame = int(math.floor(t / frame_duration))
        if on:
            sustain[start_frame:end_frame] = True
        on = (v >= threshold)
        last_time = t
    if on:
        start_frame = int(math.floor(last_time / frame_duration))
        sustain[start_frame:] = True
    return sustain


def prolong_notes_with_pedal(notes, sustain_bool, frame_duration):
    """Given pretty_midi notes modifies end times to extend while sustain_bool is True after the original end.
    Returns a list of dicts with start,end,pitch,velocity
    """
    out = []
    n_frames = len(sustain_bool)
    for n in notes:
        s = n.start
        e = n.end
        end_frame = int(math.floor(e / frame_duration))
        if end_frame < n_frames and sustain_bool[end_frame]:
            ff = end_frame
            while ff < n_frames and sustain_bool[ff]:
                ff += 1
            e = max(e, ff * frame_duration)
        out.append({'start': s, 'end': e, 'pitch': n.pitch, 'velocity': n.velocity})
    return out


def notes_to_multistate(notes_ext, total_frames, frame_duration):
    """Build multi-state (T,88) array and also collect onset-wise velocity+duration lists.
    Returns:
        states: uint8 array (T,88)
        onsets_meta: list of dicts {frame, pitch_idx, velocity, duration_seconds}
    """
    states = np.zeros((total_frames, N_KEYS), dtype=np.uint8)
    active = np.zeros((total_frames, N_KEYS), dtype=np.bool_)

    per_pitch_notes = {p: [] for p in range(PIANO_MIN, PIANO_MAX+1)}
    for n in notes_ext:
        if n['pitch'] < PIANO_MIN or n['pitch'] > PIANO_MAX:
            continue
        s_frame = int(math.floor(n['start'] / frame_duration))
        e_frame = int(math.ceil(n['end'] / frame_duration))
        s_frame = max(0, s_frame)
        e_frame = min(total_frames, e_frame)
        if s_frame >= e_frame:
            continue
        pitch_idx = n['pitch'] - PIANO_MIN
        active[s_frame:e_frame, pitch_idx] = True
        per_pitch_notes[n['pitch']].append((s_frame, e_frame, n['velocity'], n['end'] - n['start']))

    onsets_meta = []
    for pitch in range(PIANO_MIN, PIANO_MAX+1):
        idx = pitch - PIANO_MIN
        col = active[:, idx]
        if not col.any():
            continue
        for t in range(total_frames):
            if col[t]:
                if t == 0 or not col[t-1]:
                    if t > 0 and col[t-1]:
                        states[t, idx] = REONSET
                    else:
                        states[t, idx] = ONSET
                    matched = None
                    for i, (s_frame, e_frame, vel, dur_s) in enumerate(per_pitch_notes[pitch]):
                        if s_frame == t:
                            matched = per_pitch_notes[pitch].pop(i)
                            break
                    if matched is None:
                        for i, (s_frame, e_frame, vel, dur_s) in enumerate(per_pitch_notes[pitch]):
                            if s_frame <= t < e_frame:
                                matched = per_pitch_notes[pitch].pop(i)
                                break
                    if matched is not None:
                        s_frame, e_frame, vel, dur_s = matched
                        onsets_meta.append({'frame': t, 'pitch_idx': idx, 'velocity': float(vel), 'duration_s': float(dur_s)})
                else:
                    if states[t, idx] == 0:
                        states[t, idx] = SUSTAIN
            else:
                if t > 0 and col[t-1]:
                    states[t, idx] = OFFSET
    return states, onsets_meta


# ----------------------------- Main processing ----------------------------

def find_pairs(root_dir, audio_exts=('.wav', '.flac', '.mp3'), midi_exts=('.mid', '.midi')):
    """Find matched audio/midi pairs in a dataset folder. Returns list of tuples (audio_path, midi_path)
    Matching by same basename.
    """
    files = list(Path(root_dir).rglob('*'))
    audio_files = {f.stem: str(f) for f in files if f.suffix.lower() in audio_exts}
    midi_files = {f.stem: str(f) for f in files if f.suffix.lower() in midi_exts}
    common = set(audio_files.keys()).intersection(set(midi_files.keys()))
    pairs = []
    for stem in common:
        pairs.append((audio_files[stem], midi_files[stem]))
    return pairs


def process_pair(audio_path, midi_path, dataset_name, out_h5, chunk_seconds=5,
                 sustain_thresholds={'MAESTRO': 64, 'SMD': 21}):
    """Process a single audio+midi pair and write chunks to open h5py.File out_h5.
    dataset_name used to pick sustain threshold (string appears in path), fallback to MAESTRO threshold.
    """
    audio, sr_orig = sf.read(audio_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr_orig != SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr_orig, target_sr=SR)
    total_seconds = len(audio) / SR
    frame_duration = HOP / SR

    spec = compute_log_mel(audio, sr=SR)
    total_frames = spec.shape[0]

    pm = pretty_midi.PrettyMIDI(midi_path)

    events = midi_collect_control_changes(pm, control_number=64)
    thr = sustain_thresholds.get('MAESTRO')
    if 'smd' in dataset_name.lower():
        thr = sustain_thresholds.get('SMD', thr)
    elif 'maestro' in dataset_name.lower():
        thr = sustain_thresholds.get('MAESTRO', thr)
    sustain_bool = build_sustain_boolean(events, total_seconds, frame_duration, threshold=thr)

    notes = []
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append(n)
    notes_ext = prolong_notes_with_pedal(notes, sustain_bool, frame_duration)

    states_full, onsets_meta_full = notes_to_multistate(notes_ext, total_frames, frame_duration)

    chunk_samples = int(chunk_seconds * SR)
    n_samples = len(audio)
    n_chunks = int(math.ceil(n_samples / chunk_samples))

    piece_id = Path(audio_path).stem
    piece_group = out_h5.require_group(f"{dataset_name}/{piece_id}")
    piece_group.attrs['audio_path'] = audio_path
    piece_group.attrs['midi_path'] = midi_path
    piece_group.attrs['original_sr'] = sr_orig
    piece_group.attrs['resampled_sr'] = SR
    piece_group.attrs['n_frames_total'] = int(total_frames)
    piece_group.attrs['n_samples_total'] = int(n_samples)

    for ci in range(n_chunks):
        s_sample = ci * chunk_samples
        e_sample = min((ci + 1) * chunk_samples, n_samples)
        s_frame = int(np.floor(s_sample / HOP))
        e_frame = int(np.ceil(e_sample / HOP))
        if s_frame >= total_frames:
            break
        e_frame = min(e_frame, total_frames)
        spec_chunk = spec[s_frame:e_frame]
        state_chunk = states_full[s_frame:e_frame]

        onsets_chunk = [o for o in onsets_meta_full if s_frame <= o['frame'] < e_frame]
        for o in onsets_chunk:
            o['frame_local'] = int(o['frame'] - s_frame)
            o['velocity_norm'] = float(o['velocity'] / 127.0)
            o['duration_s_clipped'] = float(max(0.0, min(o['duration_s'], 5.0)))
            o['duration_norm'] = float(o['duration_s_clipped'] / 5.0)

        grp = piece_group.create_group(f"chunk_{ci:05d}")
        grp.create_dataset('spectrogram', data=spec_chunk, compression='gzip')
        grp.create_dataset('multi_state', data=state_chunk, compression='gzip', dtype='u1')
        grp.attrs['start_time_sample'] = int(s_sample)
        grp.attrs['end_time_sample'] = int(e_sample)
        grp.attrs['start_time_s'] = float(s_sample / SR)
        grp.attrs['end_time_s'] = float(e_sample / SR)

        if len(onsets_chunk) > 0:
            frames_arr = np.array([o['frame_local'] for o in onsets_chunk], dtype=np.int32)
            pitches_arr = np.array([o['pitch_idx'] for o in onsets_chunk], dtype=np.int16)
            vel_arr = np.array([o['velocity'] for o in onsets_chunk], dtype=np.int16)
            vel_norm_arr = np.array([o['velocity_norm'] for o in onsets_chunk], dtype=np.float32)
            dur_s_arr = np.array([o['duration_s_clipped'] for o in onsets_chunk], dtype=np.float32)
            dur_norm_arr = np.array([o['duration_norm'] for o in onsets_chunk], dtype=np.float32)
            grp.create_dataset('onset_frames', data=frames_arr, compression='gzip')
            grp.create_dataset('onset_pitches', data=pitches_arr, compression='gzip')
            grp.create_dataset('onset_vel', data=vel_arr, compression='gzip')
            grp.create_dataset('onset_vel_norm', data=vel_norm_arr, compression='gzip')
            grp.create_dataset('onset_dur_s', data=dur_s_arr, compression='gzip')
            grp.create_dataset('onset_dur_norm', data=dur_norm_arr, compression='gzip')
        else:
            grp.attrs['n_onsets'] = 0

    return True


def main(maestro_dir, smd_dir, out_dir, chunk_seconds=10.0):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'preprocessed_dataset.h5')
    with h5py.File(out_path, 'w') as h5f:
        if maestro_dir:
            pairs = find_pairs(maestro_dir)
            print(f"Found {len(pairs)} audio/midi pairs in MAESTRO dir")
            for a, m in tqdm(pairs, desc='MAESTRO'):
                try:
                    process_pair(a, m, 'MAESTRO', h5f, chunk_seconds=chunk_seconds)
                except Exception as e:
                    print(f"Failed {a} / {m}: {e}")
        if smd_dir:
            pairs = find_pairs(smd_dir)
            print(f"Found {len(pairs)} audio/midi pairs in SMD dir")
            for a, m in tqdm(pairs, desc='SMD'):
                try:
                    process_pair(a, m, 'SMD', h5f, chunk_seconds=chunk_seconds)
                except Exception as e:
                    print(f"Failed {a} / {m}: {e}")
    print(f"Done. Output written to {out_path}")


if __name__ == '__main__':
    # Use hardcoded paths from configuration at top
    main(MAESTRO_DIR, SMD_DIR, OUT_DIR, chunk_seconds=CHUNK_SECONDS)
