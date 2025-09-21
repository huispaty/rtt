import pretty_midi
import mir_eval
import numpy as np

def midi_to_array(midi_data):
    if isinstance(midi_data, str):
        midi_data = pretty_midi.PrettyMIDI(midi_data)
    notes = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            notes.append((note.start, note.end, note.pitch, note.velocity))
    return np.array(notes)

def compute_notewise_transcription_metrics(ref_mid, est_mid, onset_tolerance=0.05, offset_ratio=0.2, offset_min_tolerance=0.05, velocity_tolerance=0.1):
    
    if isinstance(ref_mid, str):
        ref_notes = midi_to_array(ref_mid)
    else: ref_notes = ref_mid
    if isinstance(est_mid, str):
        est_notes = midi_to_array(est_mid)
    elif isinstance(est_mid, list):
        est_notes = np.array(est_mid)
    else: est_notes = est_mid
    
    ref_intervals = ref_notes[:, :2]
    ref_pitches = ref_notes[:, 2]
    
    if isinstance(est_notes[0], dict):
        keys = ['onset_time', 'offset_time', 'midi_note', 'velocity']
        values_list = [[d[key] for key in keys] for d in est_notes]
        est_notes = np.array(values_list)
    
    est_intervals = est_notes[:, :2]
    est_pitches = est_notes[:, 2]
    
    # Compute metrics
    note_on_metrics = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=onset_tolerance,
        offset_ratio=None
    )
    
    note_on_off_metrics = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals, ref_pitches, est_intervals, est_pitches, onset_tolerance=onset_tolerance,
        offset_ratio=offset_ratio, offset_min_tolerance=offset_min_tolerance
    )

    return {
        "note-on-p": note_on_metrics[0],
        "note-on-r": note_on_metrics[1],
        "note-on-f": note_on_metrics[2],
        "note-on-off-p": note_on_off_metrics[0],
        "note-on-off-r": note_on_off_metrics[1],
        "note-on-off-f": note_on_off_metrics[2],
    }
