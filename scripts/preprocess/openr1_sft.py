# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess a chat-style dataset to parquet format.
"""

import argparse
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument(
        "--local_dataset_path",
        default=None,
        help="Local path to the raw dataset or a dataset script directory, if it exists.",
    )
    parser.add_argument(
        "--dataset_name",
        default="belati/OpenR1-Math-8192-Processed",
        help="Hugging Face dataset name (e.g., org/name) when not using --local_dataset_path.",
    )
    parser.add_argument(
        "--dataset_config",
        default="default",
        help="Dataset config name to load.",
    )
    parser.add_argument(
        "--splits",
        default="full",
        help="Comma-separated list of splits to export (e.g., 'full' or '1a,2a').",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=2780,
        help="Maximum number of samples per split (-1 for all).",
    )
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k_sft", help="The save directory for the preprocessed dataset."
    )
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path, args.dataset_config)
    else:
        dataset = datasets.load_dataset(args.dataset_name, args.dataset_config)

    if args.splits:
        requested_splits = [name.strip() for name in args.splits.split(",") if name.strip()]
        missing_splits = [name for name in requested_splits if name not in dataset]
        if missing_splits:
            raise ValueError(f"Requested splits not found: {', '.join(missing_splits)}")
        dataset = {name: dataset[name] for name in requested_splits}

    if args.max_samples is not None and args.max_samples >= 0:
        limited_dataset = {}
        for split_name, split_dataset in dataset.items():
            max_samples = min(args.max_samples, len(split_dataset))
            limited_dataset[split_name] = split_dataset.select(range(max_samples))
        dataset = limited_dataset

    # map prompt + completion into a single chat transcript
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt_messages = example.get("prompt", [])
            completion_messages = example.get("completion", [])
            messages = []
            if prompt_messages:
                messages.extend(prompt_messages)
            if completion_messages:
                messages.extend(completion_messages)
            data = {
                "messages": messages,
            }
            return data

        return process_fn

    processed_splits = {}
    for split_name, split_dataset in dataset.items():
        processed_splits[split_name] = split_dataset.map(
            function=make_map_fn(split_name),
            with_indices=True,
        )

    hdfs_dir = args.hdfs_dir

    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    for split_name, split_dataset in processed_splits.items():
        split_dataset.to_parquet(os.path.join(local_save_dir, f"{split_name}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)