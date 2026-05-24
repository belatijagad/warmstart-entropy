#!/usr/bin/env bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

shift 2

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
data_root="${repo_root}/data/openr1_sft"

torchrun --standalone --nnodes=1 --nproc_per_node="$nproc_per_node" \
    -m verl.trainer.sft_trainer \
    data.train_files="${data_root}/full.parquet" \
    data.val_files="${data_root}/full.parquet" \
    data.messages_key="messages" \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu=65536 \
    data.max_length=8192 \
    data.truncation=left \
    model.path="Qwen/Qwen2.5-3B-Instruct" \
    model.use_remove_padding=true \
    model.use_liger=true \
    model.lora_rank=128 \
    model.lora_alpha=256 \
    model.target_modules="all-linear" \
    +model.override_config.attn_implementation="flash_attention_3" \
    engine.model_dtype=bf16 \
    engine.dtype=bfloat16 \
    engine.strategy=fsdp2 \
    optim.lr=2e-4 \
    trainer.project_name="openr1-math-sft" \
    trainer.experiment_name="qwen25-3b-full" \
    trainer.total_training_steps=400 \
    trainer.total_epochs=5 \
    trainer.save_freq=15 \
    trainer.logger='[console,wandb]' \
    trainer.default_local_dir="$save_path" \
    "$@"
    