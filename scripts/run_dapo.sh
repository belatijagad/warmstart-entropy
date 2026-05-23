#!/usr/bin/env bash
set -xeuo pipefail

ray stop

export CUDA_VISIBLE_DEVICES=0
NOW=$(date +%Y%m%d)

RAY_DATA_HOME=${RAY_DATA_HOME:-"${PWD}"}
PYTHON_BIN=.venv/bin/python

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: $PYTHON_BIN not found or not executable. Create the venv first." >&2
    exit 1
fi

"$PYTHON_BIN" -m verl.trainer.main_ppo \
    --config-path="." \
    --config-name="config" \
    trainer.experiment_name="DAPO-Qwen2.5-3B-Instruct-LoRA-${NOW}-full" \
    trainer.default_local_dir="${RAY_DATA_HOME}/ckpts/multireasoner-verl/DAPO-Qwen2.5-3B-Instruct-LoRA-${NOW}-full" \
    data.train_files="${RAY_DATA_HOME}/data/dapo-math/train.parquet" \
    data.val_files="${RAY_DATA_HOME}/data/math/test.parquet"
    