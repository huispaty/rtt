from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from spec_prep import LogMelSpect
from models_init import init_layer, init_bn, init_gru, clamped_sigmoid


class CausalConv2d(nn.Conv2d):
    def forward(self, x):
        shift = self.kernel_size[1] // 2
        padding = (self.padding[0], self.padding[1] + shift)
        x = F.conv2d(x, self.weight, self.bias, self.stride,
                     padding, self.dilation, self.groups)
        if shift:
            x = x[..., :-2 * shift]
        return x


def conv_block(in_channels, out_channels, kernel_size, stride, padding, groups=1, act=nn.ReLU(), bias=False, causal=False):
    layers = OrderedDict()
    conv = CausalConv2d if causal else nn.Conv2d
    layers["conv"] = conv(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
    layers["bn"] = nn.BatchNorm2d(out_channels)
    layers["act"] = act
    init_layer(layers["conv"])
    init_bn(layers["bn"])
    return nn.Sequential(layers)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()

        C = in_channels
        r = C // reduction_ratio    # in [3_] they use 4, in [1_] they use 8

        # squeeze
        self.globpool = nn.AdaptiveAvgPool2d(output_size = (1,1)) #
        # excitation 1
        self.fc1 = nn.Linear(in_features=C, out_features=r, bias=False)
        # activation 1
        self.relu = nn.ReLU()
        # excitation 2
        self.fc2 = nn.Linear(in_features=r, out_features=C, bias=False)
        # activation 2
        self.hsigmoid = nn.Hardsigmoid() # also taken from [3_]
        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, input):
        # recalibrate feature map

        f = self.globpool(input) # [N, C, H, W]
        f = torch.flatten(f,1)
        f = self.relu(self.fc1(f))
        f = self.hsigmoid(self.fc2(f))
        f = f[:,:,None,None] # [N, C, 1, 1]

        scale = input * f
        return scale


class Residual(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class TuplePick(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]

    def extra_repr(self):
        return f"idx={self.idx}"


def mb_conv(in_channels, out_channels, kernel_size, stride, exp_factor, se=False, se_reduction=None, causal=False, act=nn.ReLU()):
    
    exp_size = in_channels * exp_factor
    padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    layers = OrderedDict()
    layers["conv_expand"] = conv_block(in_channels=in_channels,
                                       out_channels=exp_size,
                                       kernel_size=(1,1),
                                       stride=(1,1),
                                       padding=(0,0),
                                       act=act)
    layers["conv_depthwise"] = conv_block(in_channels=exp_size,
                                          out_channels=exp_size,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          groups=exp_size,
                                          padding=padding,
                                          act=act,
                                          causal=causal)
    if se:
        layers["se"] = SEBlock(in_channels=exp_size, reduction_ratio=se_reduction)
    layers["conv_project"] = conv_block(in_channels=exp_size,
                                        out_channels=out_channels,
                                        kernel_size=(1,1),
                                        stride=(1,1),
                                        padding=(0,0),
                                        act=nn.Identity())
    layers = nn.Sequential(layers)
    if in_channels == out_channels and stride == (1,1):
        layers = Residual(layers)
    return layers


def acoustic_stack(classes_num=88, mid_feat=512, momentum=0.01, se=True, causal=False, gru=True):
    
    layers = OrderedDict()
    layers["conv1"] = conv_block(in_channels=1, out_channels=32, kernel_size=(3,3), stride=(2,1), padding=(1,1), causal=causal)
    layers["drop1"] = nn.Dropout(0.2)
    layers["conv2"] = mb_conv(in_channels=32, out_channels=16, kernel_size=(3,1), stride=(1,1), exp_factor=1)
    layers["drop2"] = nn.Dropout(0.2)
    layers["mbconv3"] = mb_conv(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,1), exp_factor=6, causal=causal)
    layers["drop3"] = nn.Dropout(0.2)
    layers["mbconv4"] = mb_conv(in_channels=32, out_channels=32, kernel_size=(3,1), stride=(1,1), exp_factor=6) # res
    layers["drop4"] = nn.Dropout(0.2)
    layers["mbconv5"] = mb_conv(in_channels=32, out_channels=48, kernel_size=(3,3), stride=(2,1), exp_factor=6, causal=causal)
    layers["drop5"] = nn.Dropout(0.2)
    layers["mbconv6"] = mb_conv(in_channels=48, out_channels=48, kernel_size=(3,1), stride=(1,1), exp_factor=6) # res
    layers["drop6"] = nn.Dropout(0.2)
    layers["mbconv7"] = mb_conv(in_channels=48, out_channels=64, kernel_size=(3,3), stride=(2,1), exp_factor=6, se=se, se_reduction=8, causal=causal)
    layers["drop7"] = nn.Dropout(0.2)
    layers["mbconv8"] = mb_conv(in_channels=64, out_channels=64, kernel_size=(3,1), stride=(1,1), exp_factor=6, se=se, se_reduction=8) # res
    layers["drop8"] = nn.Dropout(0.2)
    layers["flatten9"] = Rearrange("b c f t -> b t (c f)")
    layers["fc9"] = nn.Linear(in_features=960, out_features=mid_feat, bias=False)
    layers["flip9"] = Rearrange("b t c -> b c t")
    layers["bn9"] = nn.BatchNorm1d(mid_feat, momentum=momentum)
    layers["relu9"] = nn.ReLU()
    layers["drop9"] = nn.Dropout(0.5)
    layers["flip10"] = Rearrange("b c t -> b t c")
    if gru:
        layers["gru10"] = nn.GRU(input_size=mid_feat, hidden_size=256, num_layers=2, bias=True, batch_first=True, bidirectional=False) # contains tanh non-linearity
        layers["pick10"] = TuplePick(0)
        layers["drop10"] = nn.Dropout(0.5)
        init_gru(layers["gru10"])
        layers["fc11"] = nn.Linear(in_features=256, out_features=classes_num, bias=True)
    else:
        layers["fc11"] = nn.Linear(in_features=mid_feat, out_features=classes_num, bias=True)
    init_layer(layers["fc9"])
    init_bn(layers["bn9"])
    init_layer(layers["fc11"])
    return nn.Sequential(layers)


class CustomAMT(nn.Module):
    
    def __init__(self, frames_per_second=100, classes_num=88,
                 remove_acoustic_gru=False, 
                 offset_stack=False, momentum=0.01, se=False, causal=True, 
                 clamp_sigmoid=1e-6):
        super().__init__()

        if causal and se:
            raise ValueError(f"{causal=} conflicts with {se=}.")

        sample_rate = 16000
        window_size = 2048
        hop_size = sample_rate // frames_per_second
        mel_bins = 229
        fmin = 30
        fmax = sample_rate // 2

        input_kwargs = dict(sample_rate=sample_rate, n_fft=window_size, hop_length=hop_size, f_min=fmin, f_max=fmax, n_mels=mel_bins)
        self.logmel_extractor = LogMelSpect(**input_kwargs)
        
        self.bn0 = nn.BatchNorm2d(mel_bins, momentum=momentum)

        self.frame_model = acoustic_stack(classes_num, momentum=momentum, se=se, causal=causal, gru=not remove_acoustic_gru)
        self.reg_onset_model = acoustic_stack(classes_num, momentum=momentum, se=se, causal=causal, gru=not remove_acoustic_gru)
        self.velocity_model = acoustic_stack(classes_num, momentum=momentum, se=se, causal=causal, gru=not remove_acoustic_gru)
        if offset_stack:
            self.offset_model = acoustic_stack(classes_num, momentum=momentum, se=se, causal=causal, gru=not remove_acoustic_gru)
        else:
            self.offset_model = None
        
        self.shared_model = self.frame_model[:12]
        self.frame_model = self.frame_model[12:]
        self.reg_onset_model = self.reg_onset_model[12:]
        self.velocity_model = self.velocity_model[12:]
        if self.offset_model is not None:
            self.offset_model = self.offset_model[12:]
        self.clamp_sigmoid = clamp_sigmoid

        self.frame_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=False)
        self.frame_fc = nn.Linear(256, classes_num, bias=True)

        if not offset_stack:
            self.reg_offset_gru = nn.GRU(input_size=88 * 2, hidden_size=256, num_layers=1, bias=True, batch_first=True, dropout=0., bidirectional=False)
            self.offset_fc = nn.Linear(256, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_gru(self.frame_gru)
        init_layer(self.frame_fc)
        if self.offset_model is None:
            init_gru(self.reg_offset_gru)
            init_layer(self.offset_fc)

    def forward(self, input):
        
        if input.dtype == torch.float64:
            input = input.float()
            
        x = self.logmel_extractor(input)
        x = x.unsqueeze(-1)

        x = self.bn0(x) # standardize
        x = x.transpose(1, 3).transpose(2, 3) # (batch_size, 1 , 229, T+1)

        # get outputs from acoustic stacks
        if self.shared_model is not None:
            x = self.shared_model(x)
        frame_output = self.frame_model(x)
        reg_onset_output = self.reg_onset_model(x)
        velocity_output = self.velocity_model(x)
        if self.offset_model is not None:
            offset_output = self.offset_model(x)

        reg_onset_gru_output = reg_onset_output
        
        # Use onset (and offset) to condition frames
        # clamp to prevent nans
        frame_gru_input = [frame_output,
                           reg_onset_gru_output.detach()]
        if self.offset_model is not None:
            frame_gru_input.append(offset_output.detach())
        frame_gru_input = clamped_sigmoid(torch.cat(frame_gru_input, dim=2), self.clamp_sigmoid)
        (frame_gru_output, _) = self.frame_gru(frame_gru_input)
        frame_gru_output = F.dropout(frame_gru_output, p=0.5, training=self.training, inplace=False)
        frame_gru_output = self.frame_fc(frame_gru_output)
        """(batch_size, time_steps, classes_num)"""

        # Use onset and frame to condition offset, unless we have an offset stack
        if self.offset_model is not None:
            reg_offset_gru_output = offset_output
        else:
            (reg_offset_gru_output, _) = self.reg_offset_gru(frame_gru_input)
            reg_offset_gru_output = F.dropout(reg_offset_gru_output, p=0.5, training=self.training, inplace=False)
            reg_offset_gru_output = self.offset_fc(reg_offset_gru_output)
        """(batch_size, time_steps, classes_num)"""

        output_dict = {
            'onset_output': reg_onset_gru_output,
            'offset_output': reg_offset_gru_output,
            'frame_output': frame_gru_output,
            'velocity_output': velocity_output}

        return output_dict

