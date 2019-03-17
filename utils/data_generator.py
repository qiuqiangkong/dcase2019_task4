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
    
    def __init__(self, train_hdf5_path, validate_hdf5_path, scalar, 
        batch_size, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          train_hdf5_path: string
          validate_hdf5_path: string
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
            train_dict = self.load_hdf5(train_hdf5_path)
            self.train_audio_names = train_dict['audio_name']
            self.train_features = train_dict['feature']
            
            if 'weak_target' in train_dict.keys():
                self.train_weak_targets = train_dict['weak_target']
                self.has_weak_target = True
            else:
                self.has_weak_target = False
                
            if 'strong_target' in train_dict.keys():
                self.train_strong_targets = train_dict['strong_target']
                self.has_strong_target = True
            else:
                self.has_strong_target = False
                
            self.train_index_array = np.arange(len(self.train_audio_names))
            
        # Load validation data
        validate_dict = self.load_hdf5(validate_hdf5_path)
        self.validate_audio_names = validate_dict['audio_name']
        self.validate_features = validate_dict['feature']
        self.validate_weak_targets = validate_dict['weak_target']
        self.validate_strong_targets = validate_dict['strong_target']
        self.validate_index_array = np.arange(len(self.validate_audio_names))
        
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        if train_hdf5_path:
            logging.info('Training audio num: {}'.format(len(self.train_audio_names)))            
        logging.info('Validation audio num: {}'.format(len(self.validate_audio_names)))
        
    def load_hdf5(self, hdf5_path):
        dict = {}
        
        with h5py.File(hdf5_path, 'r') as hf:
            dict['audio_name'] = np.array([name.decode() for name in hf['audio_name'][:]])

            dict['feature'] = hf['feature'][:].astype(np.float32)
            
            if 'weak_target' in hf.keys():
                dict['weak_target'] = hf['weak_target'][:].astype(np.float32)
                
            if 'strong_target' in hf.keys():
                dict['strong_target'] = hf['strong_target'][:].astype(np.float32)
            
        return dict
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing audio_name, feature, weak_target 
              and strong_target
        '''
        batch_size = self.batch_size
        indexes = np.array(self.train_index_array)
        self.random_state.shuffle(indexes)
        pointer = 0

        while True:

            # Reset pointer
            if pointer >= len(indexes):
                pointer = 0
                self.random_state.shuffle(indexes)

            # Get batch indexes
            batch_indexes = indexes[pointer: pointer + batch_size]
            pointer += batch_size

            batch_data_dict = {}

            batch_audio_name = self.train_audio_names[batch_indexes]
            batch_data_dict['audio_name'] = batch_audio_name
            
            batch_feature = self.train_features[batch_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            
            if self.has_weak_target:
                batch_weak_target = self.train_weak_targets[batch_indexes]
                batch_data_dict['weak_target'] = batch_weak_target
            
            if self.has_strong_target:
                batch_strong_target = self.train_strong_targets[batch_indexes]
                batch_data_dict['strong_target'] = batch_strong_target
            
            yield batch_data_dict
            
    def generate_validate(self, data_type, max_iteration=None):
        '''Generate mini-batch data for validation. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: None | int, use maximum iteration of partial data for
              fast evaluation
        
        Returns:
          dict containing audio_name, feature, weak_target 
              and strong_target
        '''
        batch_size = self.batch_size
        
        if data_type == 'train':
            audio_names = self.train_audio_names
            features = self.train_features
            
            if self.has_weak_target:
                weak_targets = self.train_weak_targets
                
            if self.has_strong_target:
                strong_targets = self.train_strong_targets
            
        elif data_type == 'validate':
            audio_names = self.validate_audio_names
            features = self.validate_features            
            weak_targets = self.validate_weak_targets                
            strong_targets = self.validate_strong_targets
        
        else:
            raise Exception('Incorrect argument!')
            
        audio_indexes = np.arange(len(audio_names))
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['audio_name'] = audio_names[batch_audio_indexes]
            
            batch_feature = features[batch_audio_indexes]
            batch_feature = self.transform(batch_feature)
            batch_data_dict['feature'] = batch_feature
            
            if data_type == 'validate' or self.has_weak_target:
                batch_data_dict['weak_target'] = weak_targets[batch_audio_indexes]
                
            if data_type == 'validate' or self.has_strong_target:
                batch_data_dict['strong_target'] = strong_targets[batch_audio_indexes]

            yield batch_data_dict
            
    def transform(self, x):
        return scale(x, self.scalar['mean'], self.scalar['std'])