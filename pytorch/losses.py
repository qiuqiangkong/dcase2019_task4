import torch
import torch.nn.functional as F


def to_tensor(x):
    if type(x).__name__ == 'ndarray':
        return torch.Tensor(x)
    else:
        return x


def clipwise_binary_crossentropy(output_dict, target_dict):
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['weak_target'])


def framewise_binary_crossentropy(output_dict, target_dict):
    output = output_dict['framewise_output']
    target = target_dict['strong_target']
    
    # To let output and target to have the same time steps
    N = min(output.shape[1], target.shape[1])
    
    return F.binary_cross_entropy(
        output[:, 0 : N, :], 
        target[:, 0 : N, :])


