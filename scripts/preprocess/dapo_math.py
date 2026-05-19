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
Preprocess the DAPO-Math-17k-dedup dataset to parquet format
"""

import argparse
import os

import datasets
from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--local_save_dir", default="~/data/dapo_math_17k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    data_source = "YouJiacheng/DAPO-Math-17k-dedup"
    dataset = datasets.load_dataset(data_source)

    def make_map_fn(split):
        def process_fn(example, idx):
            extra_info = {
                "split": split,
                "index": idx,
            }
            
            # Dynamically push any additional existing columns into extra_info
            for key, value in example.items():
                if key not in ["prompt", "reward_model"]:
                    extra_info[key] = value

            # Extract the ground truth (defaults to looking for "answer" or "solution")
            ground_truth = example.get("answer", example.get("solution", ""))

            data = {
                "data_source": "math_dapo",
                "prompt": example["prompt"],
                "ability": "math",
                # Updated reward_model config format for verl
                "reward_model": {
                    "ground_truth": ground_truth,
                    "style": "rule-lighteval/MATH_v2"
                },
                "extra_info": extra_info,
            }
            return data

        return process_fn

    processed_datasets = {}
    
    # Process dynamically in case there are multiple splits (e.g., train, test)
    for split in dataset.keys():
        processed_datasets[split] = dataset[split].map(
            function=make_map_fn(split), 
            with_indices=True,
            remove_columns=dataset[split].column_names # Wipes old schema to enforce the new one
        )

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir
        
    local_save_dir = os.path.expanduser(local_save_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    # Save all splits to parquet
    for split, ds in processed_datasets.items():
        ds.to_parquet(os.path.join(local_save_dir, f"{split}.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)