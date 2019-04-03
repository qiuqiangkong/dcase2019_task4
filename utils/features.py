import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], 'utils'))
import numpy as np
import argparse
import h5py
import librosa
from scipy import signal
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
import random

from utilities import (create_folder, read_audio, calculate_scalar_of_tensor, 
    pad_truncate_sequence, get_relative_path_no_extension, read_metadata, isnan)
import config


class LogMelExtractor(object):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax):
        '''Log mel feature extractor. 
        
        Args:
          sample_rate: int
          window_size: int
          hop_size: int
          mel_bins: int
          fmin: int, minimum frequency of mel filter banks
          fmax: int, maximum frequency of mel filter banks
        '''
        
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_func = np.hanning(window_size)
        
        self.melW = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=window_size, 
            n_mels=mel_bins, 
            fmin=fmin, 
            fmax=fmax).T
        '''(n_fft // 2 + 1, mel_bins)'''

    def transform(self, audio):
        '''Extract feature of a singlechannel audio file. 
        
        Args:
          audio: (samples,)
          
        Returns:
          feature: (frames_num, freq_bins)
        '''
    
        window_size = self.window_size
        hop_size = self.hop_size
        window_func = self.window_func
        
        # Compute short-time Fourier transform
        stft_matrix = librosa.core.stft(
            y=audio, 
            n_fft=window_size, 
            hop_length=hop_size, 
            window=window_func, 
            center=True, 
            dtype=np.complex64, 
            pad_mode='reflect').T
        '''(N, n_fft // 2 + 1)'''
    
        # Mel spectrogram
        mel_spectrogram = np.dot(np.abs(stft_matrix) ** 2, self.melW)
        
        # Log mel spectrogram
        logmel_spectrogram = librosa.core.power_to_db(
            mel_spectrogram, ref=1.0, amin=1e-10, 
            top_db=None)
        
        logmel_spectrogram = logmel_spectrogram.astype(np.float32)
        
        return logmel_spectrogram


def labels_to_target(labels, classes_num, lb_to_idx):
    '''Convert labels to target array. 
    E.g., ['Dog', 'Blender'] -> np.array([0, 1, 0, 0, ...])
    
    returns:
      target: (classes_num,)
    '''
    target = np.zeros(classes_num, dtype=np.bool)
    
    for label in labels:
        if not isnan(label):
            classes_id = lb_to_idx[label]
            target[classes_id] = 1
        
    return target


def events_to_target(events, frames_num, classes_num, frames_per_second, lb_to_idx):
    '''Convert events to strongly labelled matrix: (frames_num, classes_num)
    E.g., ['Dog', 'Blender'] -> np.array(
        [[0, 0, ..., 0], 
         [0, 1, ..., 0], 
         ...
         [0, 0, ..., 0]]
         
    Returns:
      target: (frames_num, classes_num)
    '''
    target = np.zeros((frames_num, classes_num), dtype=np.bool)
    
    for event_dict in events:
        if not isnan(event_dict['event']):
            class_id = lb_to_idx[event_dict['event']]
            onset_frame = int(round(event_dict['onset'] * frames_per_second))
            offset_frame = int(round(event_dict['offset'] * frames_per_second)) + 1
            target[onset_frame : offset_frame, class_id] = 1
        
    return target

 
def calculate_feature_for_all_audio_files(args):
    '''Calculate feature of audio files and write out features to a single hdf5 
    file. 
    
    Args:
      dataset_dir: string
      workspace: string
      data_type: 'development' | 'evaluation'
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arguments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    data_type = args.data_type
    mini_data = args.mini_data
    
    sample_rate = config.sample_rate
    window_size = config.window_size
    hop_size = config.hop_size
    mel_bins = config.mel_bins
    fmin = config.fmin
    fmax = config.fmax
    frames_per_second = config.frames_per_second
    frames_num = config.frames_num
    total_samples = config.total_samples
    classes_num = config.classes_num
    lb_to_idx = config.lb_to_idx
    
    # Paths    
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    relative_name = get_relative_path_no_extension(data_type)
    audios_dir = os.path.join(dataset_dir, 'audio', relative_name)

    if data_type == 'validation':
        metadata_path = os.path.join(dataset_dir, 'metadata', 'validation', 
            '{}.csv'.format(relative_name))
    else:
        metadata_path = os.path.join(dataset_dir, 'metadata', 
            '{}.csv'.format(relative_name))
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))
    create_folder(os.path.dirname(feature_path))
    
    # Feature extractor
    feature_extractor = LogMelExtractor(
        sample_rate=sample_rate, 
        window_size=window_size, 
        hop_size=hop_size, 
        mel_bins=mel_bins, 
        fmin=fmin, 
        fmax=fmax)

    # Read metadata
    (data_dict, has_weak_labels, has_strong_labels) = read_metadata(metadata_path)

    # Extract features and targets
    audio_names = sorted([*data_dict.keys()])
    
    if mini_data:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(audio_names)
        audio_names = audio_names[0 : 10]
    
    print('Extracting features of all audio files ...')
    extract_time = time.time()
    
    # Hdf5 file for storing features and targets
    hf = h5py.File(feature_path, 'w')

    hf.create_dataset(
        name='audio_name', 
        data=[audio_name.encode() for audio_name in audio_names], 
        dtype='S64')

    hf.create_dataset(
        name='feature', 
        shape=(0, frames_num, mel_bins), 
        maxshape=(None, frames_num, mel_bins), 
        dtype=np.float32)

    if has_weak_labels:
        hf.create_dataset(
            name='weak_target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype=np.bool)
            
    if has_strong_labels:
        hf.create_dataset(
            name='strong_target', 
            shape=(0, frames_num, classes_num), 
            maxshape=(None, frames_num, classes_num), 
            dtype=np.bool)
    
    for (n, audio_name) in enumerate(audio_names):
        audio_path = os.path.join(audios_dir, audio_name)
        print(n, audio_path)
        
        # Read audio
        (audio, _) = read_audio(
            audio_path=audio_path, 
            target_fs=sample_rate)
        
        # Pad or truncate audio recording
        audio = pad_truncate_sequence(audio, total_samples)
        
        # Extract feature
        feature = feature_extractor.transform(audio)
        
        # Remove the extra frames caused by padding zero
        feature = feature[0 : frames_num]
        
        hf['feature'].resize((n + 1, frames_num, mel_bins))
        hf['feature'][n] = feature
        
        if has_weak_labels:
            weak_labels = data_dict[audio_name]['weak_labels']
            hf['weak_target'].resize((n + 1, classes_num))
            hf['weak_target'][n] = labels_to_target(
                weak_labels, classes_num, lb_to_idx)
        
        if has_strong_labels:
            events = data_dict[audio_name]['strong_labels']
            hf['strong_target'].resize((n + 1, frames_num, classes_num))
            hf['strong_target'][n] = events_to_target(
                events=events, 
                frames_num=frames_num, 
                classes_num=classes_num, 
                frames_per_second=frames_per_second, 
                lb_to_idx=lb_to_idx)
            
    hf.close()
        
    print('Write hdf5 file to {} using {:.3f} s'.format(
        feature_path, time.time() - extract_time))
    
    
def calculate_scalar(args):
    '''Calculate and write out scalar. 
    
    Args:
      dataset_dir: string
      workspace: string
      data_type: 'train_weak'
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arguments & parameters
    workspace = args.workspace
    mini_data = args.mini_data
    data_type = args.data_type
    assert data_type == 'train_weak', 'We only support using train_weak data ' \
        'to calculate scalar. '
    
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    relative_name = get_relative_path_no_extension(data_type)
    
    feature_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))
        
    scalar_path = os.path.join(workspace, 'scalars', 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        '{}.h5'.format(relative_name))
        
    create_folder(os.path.dirname(scalar_path))
        
    # Load data
    load_time = time.time()
    
    with h5py.File(feature_path, 'r') as hf:
        features = hf['feature'][:]
    
    # Calculate scalar
    features = np.concatenate(features, axis=0)
    (mean, std) = calculate_scalar_of_tensor(features)
    
    with h5py.File(scalar_path, 'w') as hf:
        hf.create_dataset('mean', data=mean, dtype=np.float32)
        hf.create_dataset('std', data=std, dtype=np.float32)
    
    print('All features: {}'.format(features.shape))
    print('mean: {}'.format(mean))
    print('std: {}'.format(std))
    print('Write out scalar to {}'.format(scalar_path))
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='mode')

    # Calculate feature for all audio files
    parser_logmel = subparsers.add_parser('calculate_feature_for_all_audio_files')
    parser_logmel.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_logmel.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_logmel.add_argument('--data_type', type=str, required=True, choices=['train_weak', 'train_unlabel_in_domain', 'train_synthetic', 'validation'])
    parser_logmel.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Calculate scalar
    parser_scalar = subparsers.add_parser('calculate_scalar')
    parser_scalar.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_scalar.add_argument('--data_type', type=str, required=True, choices=['train_weak'], help='Scalar is calculated on train_weak data.')
    parser_scalar.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.mode == 'calculate_feature_for_all_audio_files':
        calculate_feature_for_all_audio_files(args)
        
    elif args.mode == 'calculate_scalar':
        calculate_scalar(args)
        
    else:
        raise Exception('Incorrect arguments!')