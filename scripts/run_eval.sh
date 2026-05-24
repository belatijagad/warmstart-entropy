#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN in your environment}"

HF_USER="${HF_USER:-belati}"
HF_REPO_PREFIX="${HF_REPO_PREFIX:-qwen25-3b}"
HF_PATH_PREFIX="${HF_PATH_PREFIX:-runs}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-Qwen/Qwen2.5-3B-Instruct}"
TASK="${TASK:-aime24_avg|0,aime25_avg|0,gsm8k|0}"
RESULTS_DIR="${RESULTS_DIR:-results}"

export HF_HUB_DISABLE_PROGRESS_BARS=1

SPLITS=(1a 2a full)
CHECKPOINT_STEPS=(15 30 45)

mkdir -p "$RESULTS_DIR"

download_checkpoint() {
  local repo_id="$1"
  local subfolder="$2"

  # Resolve HF subfolder to a local path so tokenizers load correctly.
  uv run python - "$repo_id" "$subfolder" <<'PY'
from huggingface_hub import snapshot_download
import os
import sys

repo_id = sys.argv[1]
subfolder = sys.argv[2]
cache_dir = os.environ.get("HF_CACHE_DIR") or None

path = snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    allow_patterns=[f"{subfolder}/*"],
    cache_dir=cache_dir,
)
print(os.path.join(path, subfolder))
PY
}

for split in "${SPLITS[@]}"; do
  repo_id="${HF_USER}/${HF_REPO_PREFIX}-${split}"
  repo_slug="${HF_REPO_PREFIX}-${split}"

  for step in "${CHECKPOINT_STEPS[@]}"; do
    subfolder="${HF_PATH_PREFIX}/step-${step}"
    run_name="${repo_slug}-step-${step}"
    output_dir="${RESULTS_DIR}/${run_name}"
    mkdir -p "$output_dir"

    lora_path="$(download_checkpoint "$repo_id" "${subfolder}/lora_adapter")"

    uv run lighteval vllm \
      "model_name=${BASE_MODEL_NAME},lora_path=${lora_path},dtype=bfloat16,generation_parameters={temperature:0.7,top_p:0.95,max_new_tokens:8192}" \
      "$TASK" --compute-generation-entropy --save-generations --output-dir "$output_dir"
  done
done