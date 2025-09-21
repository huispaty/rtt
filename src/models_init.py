import math
import torch
import torch.nn as nn
from scipy.special import logit

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    """Initialize a GRU layer. """
    
    def _concat_init(tensor, init_funcs):
        (length, fan_out) = tensor.shape
        fan_in = length // len(init_funcs)
    
        for (i, init_func) in enumerate(init_funcs):
            init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
        
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
    
    for i in range(rnn.num_layers):
        _concat_init(
            getattr(rnn, 'weight_ih_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, _inner_uniform]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_ih_l{}'.format(i)), 0)

        _concat_init(
            getattr(rnn, 'weight_hh_l{}'.format(i)),
            [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
        )
        torch.nn.init.constant_(getattr(rnn, 'bias_hh_l{}'.format(i)), 0)



def init_weights(model):
    """Initialize weights for all layers in the model."""
    
    def init_layer(layer):
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(layer.weight)
            if hasattr(layer, 'bias') and layer.bias is not None:
                layer.bias.data.fill_(0.)
    
    def init_bn(bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
    
    def init_gru(rnn):
        def _concat_init(tensor, init_funcs):
            (length, fan_out) = tensor.shape
            fan_in = length // len(init_funcs)
        
            for (i, init_func) in enumerate(init_funcs):
                init_func(tensor[i * fan_in : (i + 1) * fan_in, :])
            
        def _inner_uniform(tensor):
            fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
            nn.init.uniform_(tensor, -math.sqrt(3 / fan_in), math.sqrt(3 / fan_in))
        
        for i in range(rnn.num_layers):
            _concat_init(
                getattr(rnn, f'weight_ih_l{i}'),
                [_inner_uniform, _inner_uniform, _inner_uniform]
            )
            torch.nn.init.constant_(getattr(rnn, f'bias_ih_l{i}'), 0)

            _concat_init(
                getattr(rnn, f'weight_hh_l{i}'),
                [_inner_uniform, _inner_uniform, nn.init.orthogonal_]
            )
            torch.nn.init.constant_(getattr(rnn, f'bias_hh_l{i}'), 0)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            init_layer(module)
        elif isinstance(module, nn.BatchNorm2d):
            init_bn(module)
        elif isinstance(module, nn.GRU):
            init_gru(module)


def clamped_sigmoid(x, epsilon=1e-6, clamp_before=False):
    if epsilon and clamp_before:
        thresh = abs(logit(epsilon))
        x = torch.clamp(x, min=-thresh, max=thresh)
    x = torch.sigmoid(x)
    if epsilon and not clamp_before:
        x = torch.clamp(x, min=epsilon, max=1 - epsilon)
    return x



def print_model_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total model parameters {total_params:,}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable model parameters {trainable_params:,}')