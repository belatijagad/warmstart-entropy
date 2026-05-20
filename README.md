
```bash
uv sync --extra gpu --extra vllm --extra verl
```

```bash
uv run scripts/process_gsm8k.py --local_save_dir ./data/gsm8k
```

```bash
export HF_TOKEN=...
export WANDB_API=...
```

```bash
bash scripts/run_sft.sh
```

```bash
bash scripts/run_grpo.sh
```
