import os
import glob
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import librosa
import torch
 
from models import CustomAMT
from pl_model import RTT
from postprocessing import RTTPostProcessor
from evaluate import compute_notewise_transcription_metrics, midi_to_array
    
class PianoTranscription(object):
    def __init__(self, 
                 checkpoint_path='./ckpts/CustomAMT.ckpt',
                 onset_threshold=0.5,
                 offset_threshold=0.3,
                 frame_threshold=0.3,
                 overlap=True,
                 postprocessor='rtt', 
                 segment_samples=16000 * 3, 
                 device=torch.device('cuda'),                         
        ):
        
        self.segment_samples = segment_samples
        self.postprocessor = postprocessor
        self.frames_per_second = 100
        self.classes_num = 88
        self.onset_threshold = onset_threshold
        self.offset_threshold = offset_threshold
        self.frame_threshold = frame_threshold
        self.overlap = overlap
        self.device = device
        
        # Load checkpoint and model
        self.model = CustomAMT()
        rtt_model = RTT.load_from_checkpoint(model=self.model,
                                             loss_function='weighted_bce_mse', 
                                             checkpoint_path=checkpoint_path)
        self.model = rtt_model.model
        self.model.to(self.device)
        
    def transcribe(self, audio):
        
        audio = audio[None, :]
        audio_len = audio.shape[1]
        
        if self.segment_samples is not None:
            audio_before = audio
            pad_len = int(np.ceil(audio_len / self.segment_samples)) \
                * self.segment_samples - audio_len
            audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
            segments = self.enframe(audio, self.segment_samples)
            
        else: 
            segments = audio
        
        with torch.inference_mode():
            output_dict = dict()
            for segment_no in range(0, segments.shape[0], 16):
                segment_output_dict = self.model(torch.tensor(segments[segment_no : segment_no + 16]).to(self.device))
                for k in segment_output_dict.keys():
                    if k in output_dict.keys():
                        output_dict[k] = torch.cat([output_dict[k], segment_output_dict[k]], dim=0)
                    else:
                        output_dict[k] = segment_output_dict[k]
                    
        # final sigmoid
        for key in output_dict.keys():
            output_dict[key] = torch.sigmoid(output_dict[key]).cpu().numpy()
        
        if self.segment_samples is not None:
            # Deframe to original length
            for key in output_dict.keys():
                output_dict[key] = self.deframe(output_dict[key])[0 : audio_len]
            
        # postprocessing
        post_processor = RTTPostProcessor(self.frames_per_second, 
                                        self.classes_num, 
                                        onset_threshold=self.onset_threshold, 
                                        offset_threshold=self.offset_threshold, 
                                        frame_threshold=self.frame_threshold)
        
        est_note_events = post_processor.output_dict_to_midi_events(output_dict)
        
        transcribed_dict = {
            'output_dict': output_dict, 
            'est_note_events': est_note_events,
        }
        
        return transcribed_dict

    def enframe(self, x, segment_samples):
        assert x.shape[1] % segment_samples == 0
        batch = []

        pointer = 0
        while pointer + segment_samples <= x.shape[1]:
            batch.append(x[:, pointer : pointer + segment_samples])
            if self.overlap:
                pointer += segment_samples // 2 # 50% overlap 
            else:
                pointer += segment_samples

        batch = np.concatenate(batch, axis=0)
        return batch

    def deframe(self, x):
        
        if x.shape[0] == 1:
            return x[0]

        else:
            x = x[:, 0 : -1, :]
            (N, segment_samples, classes_num) = x.shape
            assert segment_samples % 4 == 0

        if self.overlap:
            y = []
            y.append(x[0, 0 : int(segment_samples * 0.75)])
            for i in range(1, N - 1):
                y.append(x[i, int(segment_samples * 0.25) : int(segment_samples * 0.75)])
            y.append(x[-1, int(segment_samples * 0.25) :])
            y = np.concatenate(y, axis=0)
            return y
        
        else:
            return x.reshape(-1, classes_num)

def inference(args):
    
    # load audio    
    sample_rate = 16000
    audio = args.audio_data
    audio_length_seconds = len(audio) / sample_rate
    
    print(f'Ref Audio: {args.audio_file_name}')
    print(f'Duration: {audio_length_seconds:.2f} seconds')
    
    # init transcriptor
    transcriptor = PianoTranscription()
    transcribed_dict = transcriptor.transcribe(audio)
    
    return transcribed_dict


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('maestro_path', type=str, help='Path to Maestro v3 dataset')
    parser.add_argument('--split', type=str, default='test')
    
    args = parser.parse_args()
    
    maestro_meta = pd.read_csv(os.path.join(args.maestro_path, 'maestro-v3.0.0.csv'))
    split_samples = maestro_meta[maestro_meta['split'] == args.split]
    
    for i, row in tqdm(split_samples.iterrows()):
        
        audio_filename = row['audio_filename'].split('/')[-1].split('.')[0]
        args.audio_file_name = audio_filename
        
        args.audio_data, _ = librosa.core.load(glob.glob(f"{args.maestro_path}/**/{row['audio_filename']}", recursive=True)[0], sr=16000, mono=True)
        
        transcribed_dict = inference(args)
        output_dict = transcribed_dict['output_dict']
        est_note_events = transcribed_dict['est_note_events']

        ref_midi_path = glob.glob(f"{args.maestro_path}/**/{row['midi_filename']}", recursive=True)[0]
        ref_note_events = midi_to_array(ref_midi_path)
                
        # evaluate
        results_csv = 'my_results.csv'
        
        metrics = dict()
        onset_offset_tolerances = [0.03, 0.02, 0.01]
        for t in onset_offset_tolerances:
            note_metrics = compute_notewise_transcription_metrics(ref_note_events, est_note_events, onset_tolerance=t)
            
            for k, v in note_metrics.items():
                metrics[f"{k}-{str(t)}"] = v
                
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df['file'] = audio_filename
        if os.path.exists(results_csv):
            metrics_df.to_csv(results_csv, mode='a', index=False, header=False)
        else:
            metrics_df.to_csv(results_csv, mode='w', index=False, header=True)

        print(f"Note-On-F 30ms: {np.round(metrics_df['note-on-f-0.03'].mean()*100, 2)}")
        print(f"Note-On-F 20ms: {np.round(metrics_df['note-on-f-0.02'].mean()*100, 2)}")
        print(f"Note-On-F 10ms: {np.round(metrics_df['note-on-f-0.01'].mean()*100, 2)}")