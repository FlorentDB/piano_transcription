import numpy as np
import librosa
from scipy.ndimage import maximum_filter

def detect_notes(C, freqs, times, 
                 threshold_db=-30, 
                 min_duration=0.08):
    """
    Detect note onsets and offsets from CQT.
    
    Parameters
    ----------
    C : complex CQT matrix (n_bins, n_frames)
    freqs : frequency array (Hz)
    times : time array (seconds)
    threshold_db : minimum dB level to consider as a note
    min_duration : minimum note duration in seconds
    
    Returns
    -------
    notes : list of dict
        Each dict contains: 'onset', 'offset', 'freq', 'midi'
    """
    # Convert to dB
    S_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)
    
    # Binary mask: where energy is above threshold
    mask = S_db > threshold_db
    
    # Local maximum filter to detect peaks in frequency
    local_max = maximum_filter(S_db, size=(3, 1)) == S_db
    peaks = mask & local_max
    
    notes = []
    min_frames = int(min_duration / (times[1] - times[0]))
    
    # Track each frequency bin
    for bin_idx in range(len(freqs)):
        active = False
        onset_frame = None
        
        for frame_idx in range(len(times)):
            if peaks[bin_idx, frame_idx] and not active:
                # Note onset
                active = True
                onset_frame = frame_idx
                
            elif active and not peaks[bin_idx, frame_idx]:
                # Note offset
                if frame_idx - onset_frame >= min_frames:
                    freq = freqs[bin_idx]
                    midi = librosa.hz_to_midi(freq)
                    
                    notes.append({
                        'onset': times[onset_frame],
                        'offset': times[frame_idx],
                        'freq': freq,
                        'midi': int(round(midi))
                    })
                active = False
        
        # Handle notes that extend to the end
        if active and len(times) - onset_frame >= min_frames:
            freq = freqs[bin_idx]
            midi = librosa.hz_to_midi(freq)
            notes.append({
                'onset': times[onset_frame],
                'offset': times[-1],
                'freq': freq,
                'midi': int(round(midi))
            })
    
    return notes


def notes_to_midi(notes, output_path='output.mid', tempo=120):
    """
    Convert detected notes to MIDI file.
    
    Parameters
    ----------
    notes : list of dict
        Notes from detect_notes()
    output_path : str
        Path to save the MIDI file
    tempo : int
        Tempo in BPM
    """
    from mido import MidiFile, MidiTrack, Message, MetaMessage
    
    if not notes:
        print("No notes to convert!")
        return
    
    # Create MIDI file
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    track.append(MetaMessage('set_tempo', tempo=int(60000000 / tempo)))
    
    # Convert notes to MIDI events (onset/offset pairs)
    events = []
    for note in notes:
        events.append({
            'time': note['onset'],
            'type': 'note_on',
            'note': note['midi'],
            'velocity': 64
        })
        events.append({
            'time': note['offset'],
            'type': 'note_off',
            'note': note['midi'],
            'velocity': 0
        })
    
    # Sort events by time
    events.sort(key=lambda x: x['time'])
    
    # Convert absolute times to delta times and add to track
    current_time = 0
    ticks_per_second = mid.ticks_per_beat * (tempo / 60)
    
    for event in events:
        # Calculate delta time in ticks
        delta_time = int((event['time'] - current_time) * ticks_per_second)
        current_time = event['time']
        
        # Add MIDI message
        if event['type'] == 'note_on':
            track.append(Message('note_on', 
                               note=event['note'], 
                               velocity=event['velocity'], 
                               time=delta_time))
        else:
            track.append(Message('note_off', 
                               note=event['note'], 
                               velocity=0, 
                               time=delta_time))
    
    # Save MIDI file
    mid.save(output_path)
    print(f"MIDI file saved to {output_path}")


def plot_piano_roll(notes, duration=None, save_path='piano_roll.png'):
    """
    Create a piano roll visualization from detected notes.
    
    Parameters
    ----------
    notes : list of dict
        Notes from detect_notes()
    duration : float or None
        Total duration in seconds (if None, use max offset)
    save_path : str
        Path to save the figure
    """
    import matplotlib.pyplot as plt
    
    if not notes:
        print("No notes to plot!")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each note as a horizontal bar
    for note in notes:
        onset = note['onset']
        offset = note['offset']
        midi = note['midi']
        duration_note = offset - onset
        
        # Draw rectangle for each note
        ax.barh(midi, duration_note, left=onset, height=0.8, 
                color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Labels and formatting
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('MIDI Note', fontsize=12)
    ax.set_title('Piano Roll - Detected Notes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits based on note range
    midi_notes = [n['midi'] for n in notes]
    min_midi = min(midi_notes) - 2
    max_midi = max(midi_notes) + 2
    ax.set_ylim(min_midi, max_midi)
    
    # Set x-axis limit
    if duration is None:
        duration = max(n['offset'] for n in notes)
    ax.set_xlim(0, duration)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Piano roll saved to {save_path}")




