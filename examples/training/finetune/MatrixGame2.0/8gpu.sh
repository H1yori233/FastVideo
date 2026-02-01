#!/bin/bash

START_FILE=${1:-1}
NODE_ID=${2:-0}

MODEL_PATH="Matrix-Game-2.0-Foundation-Diffusers"
OUTPUT_BASE="mc_32k/preprocessed"

base_port=$((29500 + NODE_ID * 100))
gpu_ids=(0 1 2 3 4 5 6 7)

for i in {1..8}; do
    port=$((base_port + i))
    gpu=${gpu_ids[((i-1))]}
    file_num=$((START_FILE + i - 1))
    
    DATA_MERGE_PATH="mc_32k/merge_${file_num}.txt"
    OUTPUT_DIR="${OUTPUT_BASE}/gpu_${gpu}_file_${file_num}"
    
    echo "Starting GPU $gpu processing file merge_${file_num}.txt on port $port"

    CUDA_VISIBLE_DEVICES=$gpu torchrun --nnodes=1 --nproc_per_node=1 --master_port $port \
        fastvideo/pipelines/preprocess/v1_preprocess.py \
        --model_path $MODEL_PATH \
        --data_merge_path $DATA_MERGE_PATH \
        --preprocess_video_batch_size 1 \
        --seed 42 \
        --max_height 352 \
        --max_width 640 \
        --num_frames 77 \
        --dataloader_num_workers 0 \
        --output_dir=$OUTPUT_DIR \
        --samples_per_file 8 \
        --train_fps 25 \
        --flush_frequency 8 \
        --preprocess_task matrixgame &
done

wait
echo "All processing blocks completed!"
