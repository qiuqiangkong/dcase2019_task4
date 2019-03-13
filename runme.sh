#!/bin/bash
# You need to modify this path to your downloaded dataset directory
DATASET_DIR='/vol/vssp/cvpnobackup/scratch_4weeks/qk00006/dcase2019/task4/dataset_root'

# You need to modify this path to your workspace to store features and models
WORKSPACE='/vol/vssp/msos/qk/workspaces/dcase2019_task4'

# Hyper-parameters
GPU_ID=1
MODEL_NAME='Cnn_9layers'
BATCH_SIZE=32

# Calculate feature
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_weak'
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_unlabel_in_domain'
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_synthetic'
python utils/features.py calculate_feature_for_all_audio_files --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='validation'

# Calculate scalar
python utils/features.py calculate_scalar --workspace=$WORKSPACE --data_type='train_weak'

# ------ Train and validate on weak labelled data ------
CUDA_VISIBLE_DEVICES=1 python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_weak' --model_name=$MODEL_NAME --loss_type='clipwise_binary_crossentropy' --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=1 python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_weak' --model_name=$MODEL_NAME --loss_type='clipwise_binary_crossentropy' --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# ------ Train and validate on synthetic data using clipwise loss ------
CUDA_VISIBLE_DEVICES=1 python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_synthetic' --model_name=$MODEL_NAME --loss_type='clipwise_binary_crossentropy' --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=1 python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_synthetic' --model_name=$MODEL_NAME --loss_type='clipwise_binary_crossentropy' --iteration=5000 --batch_size=$BATCH_SIZE --cuda

# ------ Train and validate on synthetic data using framewise loss ------
CUDA_VISIBLE_DEVICES=1 python pytorch/main.py train --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_synthetic' --model_name=$MODEL_NAME --loss_type='framewise_binary_crossentropy' --batch_size=$BATCH_SIZE --cuda

CUDA_VISIBLE_DEVICES=1 python pytorch/main.py inference_validation --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type='train_synthetic' --model_name=$MODEL_NAME --loss_type='framewise_binary_crossentropy' --iteration=5000 --batch_size=$BATCH_SIZE --cuda
