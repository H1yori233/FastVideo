#!/bin/bash
#SBATCH --job-name=mg-sf-64g
#SBATCH --partition=main
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1440G
#SBATCH --output=sf_train_output/ode_%j.out
#SBATCH --error=sf_train_output/ode_%j.err
#SBATCH --exclusive

set -e -x

# Environment Setup
source ~/conda/miniconda/bin/activate
conda activate /mnt/weka/home/hao.zhang/conda/miniconda/envs/mhuo-fv
export PYTHONPATH="/mnt/weka/home/hao.zhang/kaiqin/FastVideo_kaiqin:$PYTHONPATH"

# Basic Info
export WANDB_API_KEY="7ff8b6e8356924f7a6dd51a0342dd1a422ea9352"
export WANDB_MODE="online"
export NCCL_P2P_DISABLE=1
export MASTER_PORT=29500
export NODE_RANK=$SLURM_PROCID
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
export MASTER_ADDR=${nodes[0]}
export TOKENIZERS_PARALLELISM=false

echo "MASTER_ADDR: $MASTER_ADDR"
echo "NODE_RANK: $NODE_RANK"

# Model paths for Self-Forcing DMD distillation:
GENERATOR_MODEL_PATH="../mg_models/Matrix-Game-2.0-Student-VizDoom1k-1000steps-Diffusers"
REAL_SCORE_MODEL_PATH="../mg_models/Matrix-Game-2.0-Foundation-VizDoom1k-1000steps-Diffusers"  # Teacher model
FAKE_SCORE_MODEL_PATH="../mg_models/Matrix-Game-2.0-Foundation-VizDoom1k-1000steps-Diffusers"  # Critic model

DATA_DIR="../traindata_0209_1500/mg_ode_init/preprocessed"
VALIDATION_DATASET_FILE="examples/training/consistency_finetune/causal_matrixgame_ode_init/validation_vizdoom.json"
CKPT_SAFETENSOR="checkpoints/matrixgame_ode_init_vizdoom_new/checkpoint-6000/transformer/diffusion_pytorch_model.safetensors"

# Training arguments
training_args=(
  --tracker_project_name "matrixgame_sf_vizdoom"
  --output_dir "checkpoints/matrixgame_sf_vizdoom"
  --override_transformer_cls_name "CausalMatrixGameWanModel"
  --wandb_run_name "0211_1300_steps1500_bs64"
  --max_train_steps 1500
  --train_batch_size 1
  --train_sp_batch_size 1
  --gradient_accumulation_steps 1
  --num_latent_t 21
  --num_height 352
  --num_width 640
  --enable_gradient_checkpointing_type "full"
  --log_visualization
  --simulate_generator_forward
  --num_frames 81
  --num_frame_per_block 3  # Frame generation block size for self-forcing
  --enable_gradient_masking
  --gradient_mask_last_n_frames 21
  --init_weights_from_safetensors $CKPT_SAFETENSOR
)

# Parallel arguments
parallel_args=(
  --num_gpus 32
  --sp_size 1
  --tp_size 1
  --hsdp_replicate_dim 1
  --hsdp_shard_dim 32
)

model_args=(
  --model_path $GENERATOR_MODEL_PATH  # TODO: check if you can remove this in this script
  --pretrained_model_name_or_path $GENERATOR_MODEL_PATH
  --real_score_model_path $REAL_SCORE_MODEL_PATH
  --fake_score_model_path $FAKE_SCORE_MODEL_PATH
)

dataset_args=(
  --data_path "$DATA_DIR"
  --dataloader_num_workers 4
)

# Validation arguments
validation_args=(
  --log_validation
  --validation_dataset_file "$VALIDATION_DATASET_FILE"
  --validation_steps 100
  --validation_sampling_steps "4"
  --validation_guidance_scale "6.0"
)

# Optimizer arguments
optimizer_args=(
  --learning_rate 6e-6
  --mixed_precision "bf16"
  --weight_only_checkpointing_steps 100
  --training_state_checkpointing_steps 100
  --weight_decay 0.01
  --betas '0.0,0.999'
  --max_grad_norm 1.0
)

# Miscellaneous arguments
miscellaneous_args=(
  --inference_mode False
  --checkpoints_total_limit 3
  --training_cfg_rate 0.0
  --dit_precision "fp32"
  --flow_shift 5
  --seed 1000
  --use_ema True
  --ema_decay 0.99
  --ema_start_step 200
)

dmd_args=(
  --dmd_denoising_steps '1000,750,500,250'
  --min_timestep_ratio 0.02
  --max_timestep_ratio 0.98
  --dfake_gen_update_ratio 5
  --real_score_guidance_scale 3.0
  --fake_score_learning_rate 8e-6
  --fake_score_betas '0.0,0.999'
  --warp_denoising_step
)

self_forcing_args=(
  --independent_first_frame False  # Whether to treat first frame independently
  --same_step_across_blocks True  # Whether to use same denoising step across all blocks
  --last_step_only False  # Whether to only use the last denoising step
  --context_noise 0  # Amount of noise to add during context caching (0 = no noise)
)

mkdir -p sf_train_output

srun torchrun \
--nnodes $SLURM_JOB_NUM_NODES \
--nproc_per_node 8 \
--node_rank $SLURM_PROCID \
--rdzv_backend=c10d \
--rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
    fastvideo/training/matrixgame_self_forcing_distillation_pipeline.py \
    "${parallel_args[@]}" \
    "${model_args[@]}" \
    "${dataset_args[@]}" \
    "${training_args[@]}" \
    "${optimizer_args[@]}" \
    "${validation_args[@]}" \
    "${miscellaneous_args[@]}" \
    "${dmd_args[@]}" \
    "${self_forcing_args[@]}"
