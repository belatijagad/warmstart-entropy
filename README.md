
```bash
uv sync --extra gpu --extra vllm --extra verl
```

```bash
uv run scripts/preprocess/openr1_sft.py --local_save_dir ./data/openr1_sft --splits full --max_samples 2780
```

```bash
export HF_TOKEN=...
export WANDB_API=...
```

```bash
bash scripts/run_sft.sh
```

Checkpoints are saved to ./checkpoints/openr1-math-sft-qwen-3b-lora (set in configs/sft.yaml).

```bash
python -m verl.model_merger merge \
	--backend fsdp \
	--local_dir checkpoints/openr1-math-sft-qwen-3b-lora/global_step_400/actor \
	--target_dir ./merged/openr1-math-sft-qwen-3b-lora/step-400
```

For a single HF repo, store merged models in subfolders like runs/openr1-math-sft-qwen-3b-lora/step-400/ and track weights with Git LFS.

```bash
python scripts/upload_hf_checkpoints.py \
    --repo_id your-org/your-repo \
	--local_dir ./merged/openr1-math-sft-qwen-3b-lora/step-400 \
    --path_in_repo runs/openr1-math-sft-qwen-3b-lora/step-400
```

```bash
bash scripts/run_dapo.sh
```
