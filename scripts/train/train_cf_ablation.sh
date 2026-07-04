#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/mnt/nfs/vlm-k1kong/FastVideo}"
ENV_DIR="${ENV_DIR:-/mnt/nfs/vlm-k1kong/envs/fastvideo}"
CONFIG="${CONFIG:?Set CONFIG to a CF ablation yaml template}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to a timestamped run root}"
STAGE="${STAGE:?Set STAGE to tf or cd}"
CONDITION="${CONDITION:-$(basename "$RUN_ROOT")}"
NUM_GPUS="${NUM_GPUS:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29541}"

case "$STAGE" in
  tf|cd) ;;
  *) echo "STAGE must be 'tf' or 'cd', got '$STAGE'" >&2; exit 2 ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "Config template not found: $CONFIG" >&2
  exit 2
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Python env not found: $ENV_DIR" >&2
  exit 2
fi

STAGE_DIR="$RUN_ROOT/$STAGE"
CONFIG_DIR="$STAGE_DIR/config"
SCRIPT_DIR="$STAGE_DIR/scripts"
LOG_DIR="$STAGE_DIR/logs"
CHECKPOINT_DIR="$STAGE_DIR/checkpoints"
VALIDATION_DIR="$STAGE_DIR/validation"
TRACKER_LINK="$STAGE_DIR/tracker"
RUN_CONFIG="$CONFIG_DIR/run.yaml"
SOURCE_CONFIG="$CONFIG_DIR/source_template.yaml"
LOG_FILE="$LOG_DIR/train.log"
export RUN_CONFIG SOURCE_CONFIG

mkdir -p "$CONFIG_DIR" "$SCRIPT_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" "$VALIDATION_DIR"
ln -sfn "$CHECKPOINT_DIR/tracker" "$TRACKER_LINK"

SCRIPT_PATH="${BASH_SOURCE[0]}"
if command -v readlink >/dev/null 2>&1; then
  SCRIPT_PATH="$(readlink -f "$SCRIPT_PATH")"
fi
cp "$SCRIPT_PATH" "$SCRIPT_DIR/train_cf_ablation.sh"
cp "$CONFIG" "$SOURCE_CONFIG"

export RUN_ROOT STAGE STAGE_DIR CONDITION CHECKPOINT_DIR VALIDATION_DIR
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${STAGE}_${CONDITION}}"
export PROJECT_NAME="${PROJECT_NAME:-causal_forcing_cf_ablation}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"
export CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-}"
export CHECKPOINT_TOTAL_LIMIT="${CHECKPOINT_TOTAL_LIMIT:-}"
export VALIDATION_EVERY_STEPS="${VALIDATION_EVERY_STEPS:-}"
export GRADIENT_CHECKPOINTING_TYPE="${GRADIENT_CHECKPOINTING_TYPE:-}"
export LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:-}"
export SINK_SIZE="${SINK_SIZE:-}"

"$ENV_DIR/bin/python" - "$SOURCE_CONFIG" "$RUN_CONFIG" <<'PY'
import os
import sys
from pathlib import Path

import yaml

src, dst = sys.argv[1:3]
text = Path(src).read_text(encoding="utf-8")
for key in (
    "RUN_ROOT",
    "STAGE",
    "STAGE_DIR",
    "CONDITION",
    "CHECKPOINT_DIR",
    "VALIDATION_DIR",
    "WANDB_RUN_NAME",
    "PROJECT_NAME",
):
    text = text.replace(f"__{key}__", os.environ[key])

cfg = yaml.safe_load(text)

def set_if_env(path, env_name, cast):
    raw = os.environ.get(env_name, "")
    if raw == "":
        return
    node = cfg
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = cast(raw)

set_if_env(("training", "loop", "max_train_steps"), "MAX_TRAIN_STEPS", int)
set_if_env(("training", "checkpoint", "training_state_checkpointing_steps"), "CHECKPOINT_STEPS", int)
set_if_env(("training", "checkpoint", "checkpoints_total_limit"), "CHECKPOINT_TOTAL_LIMIT", int)
set_if_env(("callbacks", "validation", "every_steps"), "VALIDATION_EVERY_STEPS", int)
set_if_env(("pipeline", "dit_config", "local_attn_size"), "LOCAL_ATTN_SIZE", int)
set_if_env(("pipeline", "dit_config", "sink_size"), "SINK_SIZE", int)

def nullable_str(raw):
    if raw.lower() in {"none", "null", "false", "0"}:
        return None
    return raw

set_if_env(("training", "model", "enable_gradient_checkpointing_type"), "GRADIENT_CHECKPOINTING_TYPE", nullable_str)

Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

cat > "$STAGE_DIR/manifest.env" <<EOF
repo=$REPO
env_dir=$ENV_DIR
source_config=$SOURCE_CONFIG
run_config=$RUN_CONFIG
stage=$STAGE
condition=$CONDITION
run_root=$RUN_ROOT
stage_dir=$STAGE_DIR
checkpoint_dir=$CHECKPOINT_DIR
validation_dir=$VALIDATION_DIR
num_gpus=$NUM_GPUS
master_port=$MASTER_PORT
local_attn_size_override=$LOCAL_ATTN_SIZE
sink_size_override=$SINK_SIZE
gradient_checkpointing_type=${GRADIENT_CHECKPOINTING_TYPE:-default}
wandb_mode=${WANDB_MODE:-online}
wandb_api_key_set=$([[ -n "${WANDB_API_KEY:-}" ]] && echo true || echo false)
EOF

if [[ "${DRY_RUN_ONLY:-0}" == "1" ]]; then
  echo "Rendered run config: $RUN_CONFIG"
  "$ENV_DIR/bin/python" - <<'PY'
import os
from fastvideo.train.utils.config import load_run_config
cfg = load_run_config(os.environ["RUN_CONFIG"])
print("config_ok", cfg.training.tracker.run_name, cfg.training.loop.max_train_steps)
print("data_path", cfg.training.data.data_path)
print("output_dir", cfg.training.checkpoint.output_dir)
print("validation_dir", cfg.callbacks.get("validation", {}).get("output_dir"))
PY
  exit 0
fi

if [[ "${WANDB_MODE:-online}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_MODE=online requires WANDB_API_KEY in the runtime environment." >&2
  exit 2
fi

mkdir -p /mnt/lustre/vlm-k1kong/hf-cache/{hub,datasets,transformers,diffusers}
mkdir -p /mnt/lustre/vlm-k1kong/xdg-cache /mnt/lustre/vlm-k1kong/cuda-cache

export HF_HOME="${HF_HOME:-/mnt/lustre/vlm-k1kong/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/transformers}"
export DIFFUSERS_CACHE="${DIFFUSERS_CACHE:-/mnt/lustre/vlm-k1kong/hf-cache/diffusers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/mnt/lustre/vlm-k1kong/xdg-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/mnt/lustre/vlm-k1kong/cuda-cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export FASTVIDEO_ATTENTION_BACKEND="${FASTVIDEO_ATTENTION_BACKEND:-FLASH_ATTN}"
export FASTVIDEO_FLEX_ATTENTION_COMPILE_MODE="${FASTVIDEO_FLEX_ATTENTION_COMPILE_MODE:-default}"
export FASTVIDEO_DIST_TIMEOUT_MINUTES="${FASTVIDEO_DIST_TIMEOUT_MINUTES:-120}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/mnt/lustre/vlm-k1kong/triton-cache/fastvideo-cf}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/mnt/lustre/vlm-k1kong/torchinductor-cache/fastvideo-cf}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VIRTUAL_ENV="$ENV_DIR"
export PATH="$ENV_DIR/bin:$PATH"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

cd "$REPO"

cmd=(
  "$ENV_DIR/bin/python"
  -m torch.distributed.run
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --nproc_per_node "$NUM_GPUS"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  -m fastvideo.train.entrypoint.train
  --config "$RUN_CONFIG"
)

echo "CF ablation launch:"
echo "  condition: $CONDITION"
echo "  stage: $STAGE"
echo "  run root: $RUN_ROOT"
echo "  config: $RUN_CONFIG"
echo "  checkpoints: $CHECKPOINT_DIR"
echo "  validation: $VALIDATION_DIR"
echo "  log: $LOG_FILE"
echo "  GPUs: $NUM_GPUS"
echo "  W&B mode: $WANDB_MODE"
printf 'Command:'
printf ' %q' "${cmd[@]}" "$@"
echo

"${cmd[@]}" "$@" 2>&1 | tee "$LOG_FILE"
