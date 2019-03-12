import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_utils import interpolate


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2)):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=pool_size)
        
        return x
    
    
class Cnn_9layers(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_9layers, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2))
        x = self.conv_block2(x, pool_size=(2, 2))
        x = self.conv_block3(x, pool_size=(2, 2))
        tf_maps = self.conv_block4(x, pool_size=(1, 1))
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        if self.strong_target_training:
            # Clipwise prediction is obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Clipwise prediction is obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict