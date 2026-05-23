#!/usr/bin/env python3
"""Create a Hugging Face repo and upload a local folder."""

import argparse
import os

from huggingface_hub import HfApi


def parse_args():
  parser = argparse.ArgumentParser(description="Create a repo and upload a folder to Hugging Face.")
  parser.add_argument("--repo_id", required=True, help="Hugging Face repo id, e.g. org/name")
  parser.add_argument("--local_dir", required=True, help="Local directory to upload")
  parser.add_argument(
    "--path_in_repo",
    default="",
    help="Optional subfolder inside the repo (e.g., runs/exp/step-400)",
  )
  parser.add_argument("--repo_type", default="model", help="Repo type: model, dataset, or space")
  parser.add_argument("--private", action="store_true", help="Create a private repo")
  parser.add_argument(
    "--commit_message",
    default="Upload checkpoints",
    help="Commit message for the upload",
  )
  parser.add_argument(
    "--token",
    default=None,
    help="Hugging Face token (falls back to HF_TOKEN env var)",
  )
  return parser.parse_args()


def main():
  args = parse_args()
  token = args.token or os.environ.get("HF_TOKEN")
  if not token:
    raise ValueError("Missing token. Set HF_TOKEN or pass --token.")

  api = HfApi(token=token)
  api.create_repo(
    repo_id=args.repo_id,
    repo_type=args.repo_type,
    private=args.private,
    exist_ok=True,
  )
  api.upload_folder(
    repo_id=args.repo_id,
    repo_type=args.repo_type,
    folder_path=args.local_dir,
    path_in_repo=args.path_in_repo,
    commit_message=args.commit_message,
  )


if __name__ == "__main__":
  main()
