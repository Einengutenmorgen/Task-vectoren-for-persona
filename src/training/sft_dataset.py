# src/training/sft_dataset.py

from __future__ import annotations
import os 
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
load_dotenv()

os.getenv('HF_TOKEN')


@dataclass
class SFTDatasetConfig:
    """
    Configuration for loading and preparing datasets for SFT.
    """
    data_root: str
    tasks: Sequence[str]
    train_split_name: str = "train"
    eval_split_name: str = "dev"
    test_split_name: str = "test"





def _load_raw_split(cfg: SFTDatasetConfig, split_name: str) -> Dataset:
    """
    Load and concatenate the requested tasks for a given split.
    Also ensures a 'task' column is present.
    """
    root = Path(cfg.data_root)
    if not cfg.tasks:
        raise ValueError("SFTDatasetConfig.tasks must contain at least one task.")

    all_datasets = []

    for task in cfg.tasks:
        file_path = root / task / f"{split_name}.jsonl"
        
        ds = load_dataset("json", data_files=file_path, split="train")

        # Add task identifier if missing
        if "task" not in ds.column_names:
            ds = ds.map(lambda _: {"task": task})
            
        all_datasets.append(ds)

    if not all_datasets:
        raise ValueError(
            f"No data loaded for split='{split_name}' with tasks={cfg.tasks}"
        )

    return concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]


def _build_prompt_and_completion(example: dict) -> dict:
    """
    Format the example into prompt/completion pairs.
    """
    # 1. Validate presence of required fields
    required_keys = [
        "id", "task_base", "story", "question", 
        "options", "value_mapping", "task", "correct_option"
    ]
    for key in required_keys:
        if key not in example:
            raise KeyError(f"Example missing key '{key}': {example}")

    # 2. Unpack and type check
    task_base = example["task_base"]
    options = example["options"]
    value_mapping = example["value_mapping"]
    correct = example["correct_option"]

    if not isinstance(task_base, str) or len(task_base) != 2:
        raise ValueError(f"task_base must be a 2-char string: {task_base}")

    if correct not in ("A", "B"):
        raise ValueError(f"correct_option must be 'A' or 'B': {correct}")

    if not (isinstance(options, dict) and "A" in options and "B" in options):
        raise ValueError(f"Invalid options format: {options}")

    if not (isinstance(value_mapping, dict) and "A" in value_mapping and "B" in value_mapping):
        raise ValueError(f"Invalid value_mapping format: {value_mapping}")

    # 3. Build prompt
    prompt = (
        "Here is a situation that needs to be analysed. The story:\n\n"
        f"{example['story']}\n\n"
        f"Question: {example['question']}\n\n"
        "Options:\n"
        f"A. {options['A']}\n"
        f"B. {options['B']}\n\n"
        'Answer only as "A" or "B".'
    )

    return {"prompt": prompt, "completion": correct}


def load_sft_train_eval_datasets(cfg: SFTDatasetConfig) -> Tuple[Dataset, Dataset]:
    """
    Load train and dev datasets and map them to SFT format.
    """
    train_ds = _load_raw_split(cfg, cfg.train_split_name)
    eval_ds = _load_raw_split(cfg, cfg.eval_split_name)

    return (
        train_ds.map(_build_prompt_and_completion),
        eval_ds.map(_build_prompt_and_completion)
    )


def load_sft_test_dataset(cfg: SFTDatasetConfig) -> Dataset:
    """
    Load test dataset and map it to SFT format.
    """
    test_ds = _load_raw_split(cfg, cfg.test_split_name)
    return test_ds.map(_build_prompt_and_completion)