# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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
#
# Modified by the ContextPilot project in 2026.

import argparse
import logging
import os
import tempfile

import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

from verl.utils.hdfs_io import copy, makedirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_CONTENT = "You are a helpful and harmless assistant."
DEFAULT_USER_CONTENT_PREFIX = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack "
    "some knowledge, you can call a search engine by <tool_call> query </tool_call> "
    "and it will return the top searched results between <tool_response> and "
    "</tool_response>. You can search as many times as your want. If you find no "
    "further external knowledge needed, you can directly provide the answer inside "
    "<answer> and </answer>, without detailed illustrations. For example, "
    "<answer> Beijing </answer>. Question: "
)


def process_single_row(row, current_split_name, row_index):
    """
    Process a single row of data for SearchR1-like format.

    Args:
        row: DataFrame row containing the original data
        current_split_name: Name of the current split (train/test)
        row_index: Index of the row in the DataFrame

    Returns:
        pd.Series: Processed row data in the required format
    """
    question = row.get("question", "")

    user_content = user_content_prefix.rstrip("\n") + question
    prompt = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}]

    reward_model_data = row.get("reward_model")
    if isinstance(reward_model_data, dict) and "ground_truth" in reward_model_data:
        ground_truth = reward_model_data.get("ground_truth")
    else:
        ground_truth = row.get("golden_answers", [])

    data_source_tagged = "searchR1_" + str(row.get("data_source", ""))

    tools_kwargs = {
        "search": {
            "create_kwargs": {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}
        }
    }

    extra_info = {
        "index": row_index,
        "need_tools_kwargs": True,
        "question": question,
        "split": current_split_name,
        "tools_kwargs": tools_kwargs,
    }

    return pd.Series(
        {
            "data_source": data_source_tagged,
            "prompt": prompt,
            "ability": row.get("ability"),
            "reward_model": reward_model_data,
            "extra_info": extra_info,
            "metadata": row.get("metadata"),
        }
    )


def main():
    local_save_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_save_dir, exist_ok=True)

    processed_files = []

    with tempfile.TemporaryDirectory() as tmp_download_dir:
        for split in ["train", "test"]:
            parquet_filename = f"{split}.parquet"
            logger.info(f"Processing {split} split...")

            try:
                logger.info(f"Downloading {parquet_filename} from {args.hf_repo_id}")
                local_parquet_filepath = hf_hub_download(
                    repo_id=args.hf_repo_id,
                    filename=parquet_filename,
                    repo_type="dataset",
                    local_dir=tmp_download_dir,
                    local_dir_use_symlinks=False,
                )

                df_raw = pd.read_parquet(local_parquet_filepath)
                logger.info(f"Loaded {len(df_raw)} rows from {parquet_filename}")

                def apply_process_row(row, split_name=split):
                    return process_single_row(row, current_split_name=split_name, row_index=row.name)

                df_processed = df_raw.apply(apply_process_row, axis=1)

                output_file_path = os.path.join(local_save_dir, f"{split}.parquet")
                df_processed.to_parquet(output_file_path, index=False)
                logger.info(f"Saved {len(df_processed)} processed rows to {output_file_path}")
                processed_files.append(output_file_path)

            except EntryNotFoundError:
                logger.warning(f"{parquet_filename} not found in repository {args.hf_repo_id}")
            except Exception as e:
                logger.error(f"Error processing {split} split: {e}")

    if not processed_files:
        logger.warning("No data was processed or saved")
        return

    logger.info(f"Successfully processed {len(processed_files)} files to {local_save_dir}")

    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_save_dir, dst=args.hdfs_dir)
            logger.info(f"Successfully copied files to HDFS: {args.hdfs_dir}")
        except Exception as e:
            logger.error(f"Error copying files to HDFS: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Search-R1 from HuggingFace, process, and save to Parquet.")
    parser.add_argument(
        "--hf_repo_id", default="PeterJinGo/nq_hotpotqa_train", help="HuggingFace dataset repository ID."
    )
    parser.add_argument(
        "--local_dir",
        default="~/data/searchR1_processed_direct",
        help="Local directory to save the processed Parquet files.",
    )
    parser.add_argument("--hdfs_dir", default=None, help="Optional HDFS directory to copy the Parquet files to.")

    args = parser.parse_args()

    system_content = DEFAULT_SYSTEM_CONTENT
    user_content_prefix = DEFAULT_USER_CONTENT_PREFIX

    main()
