sample_rate = 32000
window_size = 1024
hop_size = 500      # So that there are 64 frames per second
mel_bins = 64
fmin = 50       # Hz
fmax = 14000    # Hz

frames_per_second = sample_rate // hop_size
audio_duration = 10     # Audio recordings in DCASE2019 Task4 are all 
                        # approximately 10 seconds
frames_num = frames_per_second * audio_duration
total_samples = sample_rate * audio_duration

# Labels
labels = ['Speech', 'Dog', 'Cat', 'Alarm_bell_ringing', 'Dishes', 'Frying', 
    'Blender', 'Running_water', 'Vacuum_cleaner', 'Electric_shaver_toothbrush']

classes_num = len(labels)
lb_to_idx = {lb: idx for idx, lb in enumerate(labels)}
idx_to_lb = {idx: lb for idx, lb in enumerate(labels)}