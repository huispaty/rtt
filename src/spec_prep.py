import torch
import torchaudio

def replace_state_dict_key(state_dict: dict, old: str, new: str):
    keys = list(state_dict.keys())  # take snapshot of the keys
    for key in keys:
        if old in key:
            state_dict[key.replace(old, new)] = state_dict.pop(key)
    return state_dict

class MelScale(torchaudio.transforms.MelScale):
    def forward(self, x, shifts=None):
        if shifts is None:
            return super().forward(x)
        else:
            factors = 2 ** (shifts / 12)
            n_stft = len(self.fb)
            fbanks = torch.stack([
                torchaudio.functional.melscale_fbanks(
                    n_stft, self.f_min * factor, self.f_max * factor,
                    self.n_mels, self.sample_rate, self.norm, self.mel_scale)
                for factor in factors]).to(x)
            
            x = x.transpose(-1, -2)
            x = torch.bmm(x, fbanks)
            x = x.transpose(-1, -2)
            return x

def asym_window(window_length, left, right):
    window = torch.ones(window_length)

    # Left taper (cosine ramp)
    if left:
        window[:left] = 0.5 * (1 - torch.cos(torch.linspace(0, torch.pi, left)))

    # Right taper (cosine ramp)
    if right:
        window[-right:] = 0.5 * (1 - torch.cos(torch.linspace(torch.pi, 0, right)))

    return window

class LogMelSpect(torch.nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=2048,
        hop_length=160,
        fft_delay=160,
        win_func='asym',
        oversampling=0,
        f_min=30,
        f_max=8000,
        n_mels=229,
        normalized="frame_length",
        power=1,
        log_multiplier=1000,
        device="cpu",
    ):
        super().__init__()
        window_fn = asym_window
        if fft_delay is not None:
            wkwargs = dict(left=n_fft - fft_delay, right=fft_delay)
        else:
            wkwargs = dict(left=n_fft // 2, right=n_fft // 2)
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft + oversampling,
            win_length=n_fft,
            hop_length=hop_length,
            normalized=normalized,
            power=power,
            center=fft_delay is None,
            window_fn=window_fn,
            wkwargs=wkwargs,
        ).to(device)
        self.fft_delay = fft_delay
        self.mel_scale = MelScale(
            n_mels,
            sample_rate,
            f_min,
            f_max,
            (n_fft + oversampling) // 2 + 1,
        ).to(device)
        self.log_multiplier = log_multiplier
        
    def forward(self, x):
        if self.fft_delay is not None:
            oversampling = self.spectrogram.n_fft - self.spectrogram.win_length
            pad_left = self.spectrogram.win_length - self.fft_delay + oversampling // 2
            pad_right = self.fft_delay + oversampling // 2
            x = torch.nn.functional.pad(x, (pad_left, pad_right))
        x = self.spectrogram(x)
        x = self.mel_scale(x, shifts=None)
        return torch.log1p(self.log_multiplier * x)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        state_dict = replace_state_dict_key(state_dict, "spect_class.", "")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
