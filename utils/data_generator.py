import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utilities import scale
import config


class DataGenerator(object):
    
    def __init__(self, train_hdf5_path, validate_hdf5_path, holdout_fold, 
        scalar, batch_size, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          train_hdf5_path: string, path of hdf5 file
          validate_hdf5_path: string, path of hdf5 file
          holdout_fold: '1' | 'none', set 1 for development and none for training 
              on all data without validation.'
          scalar: object, containing mean and std value
          batch_size: int
          seed: int, random seed
        '''
        
        self.scalar = scalar
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        
        load_time = time.time()
        
        # Load training data
        if train_hdf5_path:
            self.train_data_dict = self.load_hdf5(train_hdf5_path)            
            
        # Load validation data
        self.validate_data_dict = self.load_hdf5(validate_hdf5_path)
        
        if holdout_fold == 'none':
            self.train_data_dict, self.validate_data_dict = \
                self.combine_train_validate_data(
                    self.train_data_dict, self.validate_data_dict)
        
        self.train_audio_indexes = np.arange(
            len(self.train_data_dict['audio_name']))
            
        self.validate_audio_indexes = np.arange(len(
            self.validate_data_dict['audio_name']))        
        
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        if train_hdf5_path:
            logging.info('Training audio num: {}'.format(len(self.train_audio_indexes)))            
        logging.info('Validation audio num: {}'.format(len(self.validate_audio_indexes)))
        
        self.random_state.shuffle(self.train_audio_indexes)
        self.pointer = 0
        
        
    def load_hdf5(self, hdf5_path):
        '''Load hdf5 file. 
        
        Returns:
          data_dict: {'audio_name': (audios_num), 
                      'feature': (audios_num, frames_num, mel_bins), 
                      (if exist) 'weak_target': (audios_num, classes_num), 
                      (if exist) 'strong_target': (audios_num, frames_num, classes_num)}
        '''
        data_dict = {}

        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['audio_name'] = np.array(
                [name.decode() for name in hf['audio_name'][:]])
                
            data_dict['feature'] = hf['feature'][:].astype(np.float32)
            
            if 'weak_target' in hf.keys():
                data_dict['weak_target'] = \
                    hf['weak_target'][:].astype(np.float32)
                
            if 'strong_target' in hf.keys():
                data_dict['strong_target'] = \
                    hf['strong_target'][:].astype(np.float32)
            
        return data_dict
        
    def combine_train_validate_data(self, train_data_dict, validate_data_dict):
        '''Combine train and validate data to for training on full data. 
        '''
        new_train_data_dict = {}
        new_validate_data_dict = {}
        
        for key in train_data_dict.keys():
            new_train_data_dict[key] = np.concatenate(
                (train_data_dict[key], validate_data_dict[key]), axis=0)
        
            new_validate_data_dict[key] = []
        
        return new_train_data_dict, new_validate_data_dict
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: e.g.:
              {'audio_name': (batch_size), 
               'feature': (batch_size, frames_num, mel_bins), 
               (if exist) 'weak_target': (batch_size, classes_num), 
               (if exist) 'strong_target': (batch_size, frames_num, classes_num)}
        '''
        while True:

            # Reset pointer
            if self.pointer >= len(self.train_audio_indexes):
                self.pointer = 0
                self.random_state.shuffle(self.train_audio_indexes)

            # Get batch indexes
            batch_indexes = self.train_audio_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                self.train_data_dict['audio_name'][batch_indexes]
            
            batch_feature = self.train_data_dict['feature'][batch_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            
            if 'weak_target' in self.train_data_dict:
                batch_data_dict['weak_target'] = \
                    self.train_data_dict['weak_target'][batch_indexes]
            
            if 'strong_target' in self.train_data_dict:
                batch_data_dict['strong_target'] = \
                    self.train_data_dict['strong_target'][batch_indexes]
            
            yield batch_data_dict
            
    def generate_validate(self, data_type, max_iteration=None):
        '''Generate mini-batch data for validation. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: None | int, use maximum iteration of partial data for
              fast evaluation
        
        Returns:
          batch_data_dict: e.g.:
              {'audio_name': (batch_size), 
               'feature': (batch_size, frames_num, mel_bins), 
               (if exist) 'weak_target': (batch_size, classes_num), 
               (if exist) 'strong_target': (batch_size, frames_num, classes_num)}
        '''
        
        if data_type == 'train':
            data_dict = self.train_data_dict            
        elif data_type == 'validate':
            data_dict = self.validate_data_dict
        else:
            raise Exception('Incorrect argument!')
            
        audios_num = len(data_dict['audio_name'])
        audio_indexes = np.arange(audios_num)
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= audios_num:
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[
                pointer: pointer + self.batch_size]
                
            pointer += self.batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = \
                data_dict['audio_name'][batch_audio_indexes]
            
            batch_feature = data_dict['feature'][batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            
            if 'weak_target' in data_dict:
                batch_data_dict['weak_target'] = \
                    data_dict['weak_target'][batch_audio_indexes]
                
            if 'strong_target' in data_dict:
                batch_data_dict['strong_target'] = \
                    data_dict['strong_target'][batch_audio_indexes]

            yield batch_data_dict
            
    def transform(self, x):
        return scale(x, self.scalar['mean'], self.scalar['std'])