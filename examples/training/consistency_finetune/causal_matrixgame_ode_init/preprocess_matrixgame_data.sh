#!/bin/bash

GPU_NUM=1 # 2,4,8
MODEL_PATH="Matrix-Game-2.0-Foundation-Diffusers"
DATA_MERGE_PATH="footsies-dataset/merge.txt"
OUTPUT_DIR="footsies-dataset/ode-init/"

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/pipelines/preprocess/v1_preprocess.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --preprocess_video_batch_size 1 \
    --seed 42 \
    --max_height 350 \
    --max_width 640 \
    --num_frames 77 \
    --flow_shift 5.0 \
    --dataloader_num_workers 0 \
    --output_dir=$OUTPUT_DIR \
    --train_fps 25 \
    --samples_per_file 8 \
    --flush_frequency 8 \
    --video_length_tolerance_range 5 \
    --preprocess_task "matrixgame_ode_trajectory" 
