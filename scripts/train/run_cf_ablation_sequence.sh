#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
EXP_ROOT="${EXP_ROOT:-/mnt/lustre/vlm-k1kong/experiments/cf_ablation}"
CONFIG_DIR="${CONFIG_DIR:-examples/train/configs/cf_pipeline/ablation}"
PROJECT_NAME="${PROJECT_NAME:-causal_forcing_cf_ablation}"
WANDB_MODE="${WANDB_MODE:-online}"
NUM_GPUS="${NUM_GPUS:-4}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29700}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-relrope}"

if [[ "$WANDB_MODE" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in the runtime environment." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/mnt/lustre/vlm-k1kong/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/diffusers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/lustre/vlm-k1kong/xdg-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/lustre/vlm-k1kong/cuda-cache}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export ROPE_CACHE_POLICY="${ROPE_CACHE_POLICY:-relativistic}"
export FASTVIDEO_DIST_TIMEOUT_MINUTES="${FASTVIDEO_DIST_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/lustre/vlm-k1kong/triton-cache/fastvideo-cf}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/lustre/vlm-k1kong/torchinductor-cache/fastvideo-cf}"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

mkdir -p "$EXP_ROOT" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$DIFFUSERS_CACHE" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR"

sequence_ts=$(date +%Y%m%d_%H%M%S)
sequence_dir="$EXP_ROOT/full_sequence_${sequence_ts}"
mkdir -p "$sequence_dir"
sequence_log="$sequence_dir/sequence.log"
runs_tsv="$sequence_dir/runs.tsv"
printf '%s\n' "$sequence_dir" > "$EXP_ROOT/FULL_SEQUENCE_LATEST"
printf 'condition\trun_label\trun_root\n' > "$runs_tsv"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$sequence_log"
}

cd "$REPO"
log "Starting CF ablation sequence in $sequence_dir"
log "Conditions: ${RUN_CONDITIONS:-sink0_local6 sink0_local9 sink1_local6 sink1_local9}"
log "Run name suffix: $RUN_NAME_SUFFIX"
log "RoPE cache policy: $ROPE_CACHE_POLICY"

idx=0
for condition in ${RUN_CONDITIONS:-sink0_local6 sink0_local9 sink1_local6 sink1_local9}; do
  run_ts=$(date +%Y%m%d_%H%M%S)
  run_label="${condition}_${RUN_NAME_SUFFIX}"
  run_root="$EXP_ROOT/${run_ts}_${run_label}_tf3k_cd2k"
  printf '%s\t%s\t%s\n' "$condition" "$run_label" "$run_root" >> "$runs_tsv"
  log "Condition $condition: run_label=$run_label run_root=$run_root"

  tf_port=$((MASTER_PORT_BASE + idx * 2))
  cd_port=$((MASTER_PORT_BASE + idx * 2 + 1))

  log "Condition $condition: TF start"
  CONFIG="$CONFIG_DIR/tf_${condition}.yaml" \
  RUN_ROOT="$run_root" \
  STAGE=tf \
  CONDITION="$run_label" \
  NUM_GPUS="$NUM_GPUS" \
  MASTER_PORT="$tf_port" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_RUN_NAME="tf_${run_label}_${run_ts}" \
  PROJECT_NAME="$PROJECT_NAME" \
  bash scripts/train/train_cf_ablation.sh
  log "Condition $condition: TF complete"

  log "Condition $condition: export TF checkpoint"
  mkdir -p "$run_root/export/tf"
  "$ENV_DIR/bin/python" -m fastvideo.train.entrypoint.dcp_to_diffusers \
    --role student \
    --checkpoint "$run_root/tf/checkpoints" \
    --output-dir "$run_root/export/tf" \
    --overwrite 2>&1 | tee "$run_root/export/tf/export.log"
  test -f "$run_root/export/tf/transformer/model.safetensors"
  log "Condition $condition: export complete"

  log "Condition $condition: CD start"
  CONFIG="$CONFIG_DIR/cd_${condition}.yaml" \
  RUN_ROOT="$run_root" \
  STAGE=cd \
  CONDITION="$run_label" \
  NUM_GPUS="$NUM_GPUS" \
  MASTER_PORT="$cd_port" \
  WANDB_MODE="$WANDB_MODE" \
  WANDB_RUN_NAME="cd_${run_label}_${run_ts}" \
  PROJECT_NAME="$PROJECT_NAME" \
  bash scripts/train/train_cf_ablation.sh
  log "Condition $condition: CD complete"

  idx=$((idx + 1))
done

log "CF ablation sequence complete"
