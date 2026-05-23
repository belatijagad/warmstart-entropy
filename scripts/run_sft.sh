#!/usr/bin/env bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

shift 2

torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
    -m verl.trainer.sft_trainer \
    --config-path="." \
    --config-name="sft_config" \
    trainer.default_local_dir="$save_path" \
    "$@"
    