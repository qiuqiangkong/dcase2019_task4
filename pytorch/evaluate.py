import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))

import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import sed_eval

from utilities import get_filename, read_csv_file_for_sed_eval_tool, inverse_scale
from pytorch_utils import forward
from vad import activity_detection
import config


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True, verbose=False):
        '''Evaluator to write out submission and evaluate performance. 
        
        Args:
          model: object
          data_generator: object
          cuda: bool
        '''
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        self.verbose = verbose
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        
        # Hyper-parameters for predicting events from framewise predictions
        self.audio_tagging_thres = 0.5
        self.sed_high_thres = 0.9
        self.sed_low_thres = 0.5
        self.n_smooth = self.frames_per_second // 4
        self.n_salt = self.frames_per_second // 4

    def evaluate(self, data_type, metadata_path, submission_path, 
        max_iteration=None):
        '''Write out submission file and evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          metadata_path: string
          submission_path: string
          max_iteration: None | int, maximum iteration to run to speed up evaluation
        '''
        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
            
        # Evaluate audio tagging
        if 'weak_target' in output_dict:
            weak_target = output_dict['weak_target']
            clipwise_output = output_dict['clipwise_output']
            ap_array = metrics.average_precision_score(
                weak_target, clipwise_output, average=None)
            mAP = np.mean(ap_array)
            
            logging.info('{} statistics:'.format(data_type))       
            logging.info('    Audio tagging mAP: {:.3f}'.format(mAP))
            
        # Write out submission file and evaluate SED with official tools
        if 'strong_target' in output_dict:
            audio_names = output_dict['audio_name']
            clipwise_output = output_dict['clipwise_output']
            framewise_output = output_dict['framewise_output']
            
            (audios_num, frames_num, classes_num) = framewise_output.shape
            
            f = open(submission_path, 'w')
            f.write('{}\t{}\t{}\t{}\n'.format(
                'filename', 'onset', 'offset', 'event_label'))
            
            for n in range(audios_num):
                for k in range(classes_num):
                    if clipwise_output[n, k] > self.audio_tagging_thres:
                        bgn_fin_pairs = activity_detection(
                            x=output_dict['framewise_output'][n, :, k], 
                            thres=self.sed_high_thres, 
                            low_thres=self.sed_low_thres, 
                            n_smooth=self.n_smooth, 
                            n_salt=self.n_salt)
                        
                        for pair in bgn_fin_pairs:
                            bgn_time = pair[0] / float(self.frames_per_second)
                            fin_time = pair[1] / float(self.frames_per_second)
                            f.write('{}\t{}\t{}\t{}\n'.format(
                                audio_names[n], bgn_time, fin_time, self.labels[k]))
            f.close()
                        
            logging.info('    Write submission file to {}'.format(submission_path))
            
            reference_dict = read_csv_file_for_sed_eval_tool(metadata_path)
            predict_dict = read_csv_file_for_sed_eval_tool(submission_path)

            # Event based metrics
            event_based_metric = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=config.labels, 
                evaluate_onset=True,
                evaluate_offset=True,
                t_collar=0.200,
                percentage_of_length=0.2)
            
            # Segment based metrics
            segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=config.labels, 
                time_resolution=0.2)
            
            for audio_name in audio_names:
                if audio_name in reference_dict.keys():
                    ref_list = reference_dict[audio_name]
                else:
                    ref_list = []
                    
                if audio_name in predict_dict.keys():
                    pred_list = predict_dict[audio_name]
                else:
                    pred_list = []
                    
                event_based_metric.evaluate(ref_list, pred_list)
                segment_based_metric.evaluate(ref_list, pred_list)
                    
            event_metrics = event_based_metric.results_class_wise_average_metrics()
            logging.info('    Event-based, classwise F score: {:.3f}, ER: {:.3f}, Del: {:.3f}, Ins: {:.3f}'.format(
                event_metrics['f_measure']['f_measure'], event_metrics['error_rate']['error_rate'], 
                event_metrics['error_rate']['deletion_rate'], event_metrics['error_rate']['insertion_rate']))

            segment_metrics = segment_based_metric.results_class_wise_average_metrics()
            logging.info('    Segment based, classwise F score: {:.3f}, ER: {:.3f}, Del: {:.3f}, Ins: {:.3f}'.format(
                segment_metrics['f_measure']['f_measure'], segment_metrics['error_rate']['error_rate'], 
                segment_metrics['error_rate']['deletion_rate'], segment_metrics['error_rate']['insertion_rate']))
                
            if self.verbose:
                logging.info(event_based_metric)
                logging.info(segment_based_metric)        
            
    def visualize(self, data_type):
        '''Visualize logmel spectrogram, reference and prediction. 
        
        Args:
          data_type: 'train' | 'validate'
        '''
        
        mel_bins = config.mel_bins
        audio_duration = config.audio_duration
        frames_num = config.frames_num
        classes_num = config.classes_num
        labels = config.labels
        
        # Forward
        generate_func = self.data_generator.generate_validate(
            data_type=data_type)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_input=True, 
            return_target=True)

        for n, audio_name in enumerate(output_dict['audio_name']):
            
            print('File: {}'.format(audio_name))
            for k in range(classes_num):
                print('{:<20}{:<8}{:.3f}'.format(labels[k], 
                    output_dict['weak_target'][n, k], output_dict['clipwise_output'][n, k]))
                
            event_prediction = np.zeros((frames_num, classes_num))
                
            for k in range(classes_num):
                if output_dict['clipwise_output'][n, k] > self.audio_tagging_thres:
                    bgn_fin_pairs = activity_detection(
                        x=output_dict['framewise_output'][n, :, k], 
                        thres=self.sed_high_thres, 
                        low_thres=self.sed_low_thres, 
                        n_smooth=self.n_smooth, 
                        n_salt=self.n_salt)
                    
                    for pair in bgn_fin_pairs:
                        event_prediction[pair[0] : pair[1], k] = 1
            
            fig, axs = plt.subplots(4, 1, figsize=(10, 8))
            logmel = inverse_scale(output_dict['feature'][n], 
                self.data_generator.scalar['mean'], 
                self.data_generator.scalar['std'])
            axs[0].matshow(logmel.T, origin='lower', aspect='auto', cmap='jet')
            if 'strong_target' in output_dict.keys():
                axs[1].matshow(output_dict['strong_target'][n].T, origin='lower', aspect='auto', cmap='jet')
            masked_framewise_output = output_dict['framewise_output'][n] * output_dict['clipwise_output'][n]
            axs[2].matshow(masked_framewise_output.T, origin='lower', aspect='auto', cmap='jet')
            axs[3].matshow(event_prediction.T, origin='lower', aspect='auto', cmap='jet')
            
            axs[0].set_title('Log mel spectrogram', color='r')
            axs[1].set_title('Reference sound events', color='r')
            axs[2].set_title('Framewise prediction', color='b')
            axs[3].set_title('Eventwise prediction', color='b')
            
            for i in range(4):
                axs[i].set_xticks([0, frames_num])
                axs[i].set_xticklabels(['0', '{:.1f} s'.format(audio_duration)])
                axs[i].xaxis.set_ticks_position('bottom')
                axs[i].set_yticks(np.arange(classes_num))
                axs[i].set_yticklabels(labels)
                axs[i].yaxis.grid(color='w', linestyle='solid', linewidth=0.2)
            
            axs[0].set_ylabel('Mel bins')
            axs[0].set_yticks([0, mel_bins])
            axs[0].set_yticklabels([0, mel_bins])
            
            fig.tight_layout()
            plt.show()
            