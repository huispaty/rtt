import io
import numpy as np

class RTTPostProcessor(object):
    def __init__(self, 
                 frames_per_second, 
                 classes_num, 
                 onset_threshold,
                 offset_threshold,
                 frame_threshold):
        
        self.frames_per_second = frames_per_second
        self.classes_num = classes_num
        
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold 
        self.frame_threshold = frame_threshold
        
        self.begin_note = 21
        self.velocity_scale = 127

    def output_dict_to_midi_events(self, output_dict):
        """Main function. Post process model outputs to MIDI events.

        Args:
          output_dict: {
            'onset_output': (segment_frames, classes_num), 
            'offset_output': (segment_frames, classes_num), 
            'frame_output': (segment_frames, classes_num), 
            'velocity_output': (segment_frames, classes_num), 

        Outputs:
          est_note_events: list of dict, e.g. [
            {'onset_time': 39.74, 'offset_time': 39.87, 'midi_note': 27, 'velocity': 83}, 
            {'onset_time': 11.98, 'offset_time': 12.11, 'midi_note': 33, 'velocity': 88}]

        """
        ############################ Binarize predicted piano rolls
        onset_output = output_dict['onset_output'] # (times, pitches)
        frame_output = output_dict['frame_output']
        velocity_output = output_dict['velocity_output']
        offset_output = output_dict['offset_output']
        # onsets
        onsets_binarized = (onset_output >= self.onset_threshold).astype(int)
        # take first frame as is, for the remainder onset if the previous frame was below the threshold (0) and the current is above (1)
        onsets_binarized = np.concatenate((onsets_binarized[:1, :], onsets_binarized[1:, :] & ~onsets_binarized[:-1, :]), axis=0)
        
        # offsets
        offsets_binarized = (offset_output >= self.offset_threshold).astype(int)
        # check if the last frames offset was above threshold, if so keep it
        # deactivate offset frames where an onset is active
        offsets_binarized = np.where(onsets_binarized, 0, offsets_binarized)
        # keep only offsets for which the last frame was below the threshold and the current one is above
        offsets_binarized = np.where(offsets_binarized > np.roll(offsets_binarized, shift=1, axis=0), 1, 0)
        # deactivate offsets for the first frame
        offsets_binarized[0,:] = 0
        # frames
        frames_binarized = (frame_output >= self.frame_threshold).astype(int)
        frames_binarized = np.where(
            # mask: if onset or previous frame was on, continue with frame
            (onsets_binarized | np.roll(frames_binarized, shift=1, axis=0)),
            frames_binarized,
            0        
        )
        # velocities
        velocities = velocity_output * onsets_binarized
        # velocity scaling from onsets and frames
        velocities = np.where(velocities > 0, velocities*self.velocity_scale, 0)
        velocities = velocities.astype(int) # (times, pitches)
        
        # # ############################ Create note tuples from binarized rolls
        
        note_tuples = []
        time_steps, pitches = onsets_binarized.shape
        # keep count of active pitches and their last onset times
        active_pitches = np.zeros(pitches, dtype=int)
        active_pitches_start_frame = np.zeros(pitches, dtype=int)
        midi_pitches = np.arange(self.begin_note, pitches+self.begin_note)
        max_note_duration = 300 # 3 sec
        min_note_duration = 2 # 20ms
        min_reonset_gap = min_note_duration + 1 # at least 30ms between reonsets of the same pitch
        
        for t in range(time_steps):
            # check for new onsets
            pitch_onsets = np.where(onsets_binarized[t] == 1)[0]
            offsets_in_current_frame = np.zeros(pitches, dtype=int)
            reonset_pitches = np.zeros(pitches, dtype=int)
            
            # for each active pitch, check if it's already in active pitches
            for pitch in pitch_onsets:
                
                # it's not yet in active pitches, add the onset time
                if active_pitches[pitch] == 0: 
                    active_pitches[pitch] = 1
                    active_pitches_start_frame[pitch] = t
                else:
                    # reonset
                    if t - active_pitches_start_frame[pitch] >= min_reonset_gap:
                        offsets_in_current_frame[pitch] = 1
                        reonset_pitches[pitch] = 1
        
            # check for frame offsets
            inactive_frame_pitches = np.where(frames_binarized[t] == 0)[0]
            for pitch in inactive_frame_pitches:
                # pitch is found in active pitches
                if active_pitches[pitch] == 1:
                    if t - active_pitches_start_frame[pitch] >= min_note_duration:
                        offsets_in_current_frame[pitch] = 1
            
            # check for frame offsets
            offsets_detected_pitches = np.where(offsets_binarized[t] == 1)[0]
            for pitch in offsets_detected_pitches:
                # pitch is found in active pitches
                if active_pitches[pitch] == 1:
                    if t - active_pitches_start_frame[pitch] >= min_note_duration:
                        offsets_in_current_frame[pitch] = 1
            
            # check for long notes
            for pitch in np.where(active_pitches == 1)[0]:
                if t - active_pitches_start_frame[pitch] >= max_note_duration:
                    offsets_in_current_frame[pitch] = 1
                
            # decode offsets into notes    
            for pitch in np.where(offsets_in_current_frame == 1)[0]:
                pitch_start = active_pitches_start_frame[pitch]
                pitch_end = t
                note_tuple = (pitch_start, pitch_end, midi_pitches[pitch], int(velocities[pitch_start, pitch]))
                note_tuples.append(note_tuple)
                active_pitches_start_frame[pitch] = 0
                active_pitches[pitch] = 0
                
                if reonset_pitches[pitch] == 1:
                    active_pitches[pitch] = 1
                    active_pitches_start_frame[pitch] = t
        
        est_midi_notes = np.array(note_tuples)
        
        if len(est_midi_notes) == 0:
            return []
        
        onset_times = est_midi_notes[:, 0] / self.frames_per_second
        offset_times = est_midi_notes[:, 1] / self.frames_per_second
        pitches = est_midi_notes[:, 2]
        velocities = est_midi_notes[:, 3]
        
        est_on_off_note_vels = np.stack((onset_times, offset_times, pitches, velocities), axis=-1)
        est_on_off_note_vels = est_on_off_note_vels.astype(np.float32)        
        
        # ############################ Convert to MIDI events
        midi_events = []
        for i in range(len(est_on_off_note_vels)):
            midi_events.append({
                'onset_time': est_on_off_note_vels[i][0], 
                'offset_time': est_on_off_note_vels[i][1], 
                'midi_note': int(est_on_off_note_vels[i][2]), 
                'velocity': int(est_on_off_note_vels[i][3])})
            
            # TMP change its too late
            # midi_events.append((est_on_off_note_vels[i][0], est_on_off_note_vels[i][1], int(est_on_off_note_vels[i][2]), int(est_on_off_note_vels[i][3])))

        return midi_events
