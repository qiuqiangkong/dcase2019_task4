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
    
    
class Cnn_5layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num, strong_target_training):
        super(Cnn_5layers_AvgPooling, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
                              
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                              kernel_size=(5, 5), stride=(1, 1),
                              padding=(2, 2), bias=False)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.fc)
        
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 8
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        
        x = F.relu_(self.bn4(self.conv4(x)))
        tf_maps = F.avg_pool2d(x, kernel_size=(1, 1))
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        framewise_vector = torch.mean(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
    

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
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_AvgPooling(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_9layers_AvgPooling, self).__init__()

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
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        tf_maps = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
        
        
class Cnn_9layers_MaxPooling(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_9layers_MaxPooling, self).__init__()

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
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='max')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='max')
        tf_maps = self.conv_block4(x, pool_size=(1, 1), pool_type='max')
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict
        
        
class Cnn_13layers_AvgPooling(nn.Module):
    def __init__(self, classes_num, strong_target_training=False):
        
        super(Cnn_13layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()
        
        self.strong_target_training = strong_target_training

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        interpolate_ratio = 32
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        tf_maps = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        '''Time-frequency maps: (batch_size, channels_num, times_steps, freq_bins)'''

        (framewise_vector, _) = torch.max(tf_maps, dim=3)
        '''(batch_size, feature_maps, frames_num)'''
        
        output_dict = {}
        
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output

        # Framewise prediction
        framewise_output = torch.sigmoid(self.fc(framewise_vector.transpose(1, 2)))
        framewise_output = interpolate(framewise_output, interpolate_ratio)
        '''(batch_size, frames_num, classes_num)'''
            
        output_dict['framewise_output'] = framewise_output
        
        # Clipwise prediction
        if self.strong_target_training:
            # Obtained by taking the maximum framewise predictions
            (output_dict['clipwise_output'], _) = torch.max(framewise_output, dim=1)
            
        else:
            # Obtained by applying fc layer on aggregated framewise_vector
            (aggregation, _) = torch.max(framewise_vector, dim=2)
            output_dict['clipwise_output'] = torch.sigmoid(self.fc(aggregation))

        return output_dict