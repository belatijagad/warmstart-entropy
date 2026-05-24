#!/usr/bin/env bash
set -euo pipefail

# Merge FSDP checkpoints with verl and upload to Hugging Face.
# Edit the variables below before running.

HF_ORG="belati"
OUTPUT_ROOT="outputs"
MERGED_ROOT="merged"
SPLITS=("1a" "2a" "full")
STEPS=(15 30 45)
HF_PATH_PREFIX="runs"

# Optional: set HF_TOKEN in env or export it before running.
: "${HF_TOKEN:=}"

if [[ -z "${HF_ORG}" ]]; then
  echo "Please set HF_ORG in scripts/merge_and_upload.sh" >&2
  exit 1
fi

for split in "${SPLITS[@]}"; do
  for step in "${STEPS[@]}"; do
    local_dir="${OUTPUT_ROOT}/qwen25-3b-${split}/global_step_${step}"
    target_dir="${MERGED_ROOT}/qwen25-3b-${split}/step-${step}"

    if [[ ! -d "${local_dir}" ]]; then
      echo "Skipping missing ${local_dir}" >&2
      continue
    fi

    echo "Merging ${local_dir} -> ${target_dir}"
    uv run -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${local_dir}" \
      --target_dir "${target_dir}"

    repo_id="${HF_ORG}/qwen25-3b-${split}"
    path_in_repo="${HF_PATH_PREFIX}/step-${step}"
    echo "Uploading ${target_dir} -> ${repo_id}:${path_in_repo}"
    python scripts/upload_hf_checkpoints.py \
      --repo_id "${repo_id}" \
      --local_dir "${target_dir}" \
      --path_in_repo "${path_in_repo}"
  done
done
