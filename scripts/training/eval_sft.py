# scripts/training/eval_sft.py

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from training.sft_dataset import SFTDatasetConfig, load_sft_test_dataset


# =========================
# EXPLICIT CONFIGURATION
# =========================

# MUST be set before running.
MODEL_NAME_OR_PATH: str | None = None  # e.g. "/models/llama32_1b_sft_ab_bc_ca/merged"
DATA_ROOT: str | None = None           # same as training data root

# Only evaluate on AB, BC, CA as per your scope.
TASKS = ("AB", "BC", "CA")

# Evaluation options.
MAX_SEQ_LENGTH: int = 512
MAX_NEW_TOKENS: int = 2

# Device must be specified explicitly; no silent fallback.
DEVICE: str = "cuda:0"  # e.g. "cuda:0" or "cpu"


def _validate_config() -> None:
    if not MODEL_NAME_OR_PATH:
        raise ValueError("MODEL_NAME_OR_PATH must be set to the trained (merged) model directory.")
    if not DATA_ROOT:
        raise ValueError("DATA_ROOT must be set to the dataset root directory.")
    if not TASKS:
        raise ValueError("TASKS must contain at least one task.")

    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"DEVICE={DEVICE} requested but CUDA is not available.")


def _extract_choice_from_text(text: str) -> str:
    """
    Extract the first 'A' or 'B' from generated text.
    Fail fast if neither is found.
    """
    for ch in text:
        if ch in ("A", "B"):
            return ch
    raise RuntimeError(f"Model output does not contain 'A' or 'B': {repr(text)}")


def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_seq_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """
    Run greedy evaluation and compute overall + per-task_base accuracy.
    """
    model.eval()
    model.to(device)

    total = 0
    correct = 0

    per_task_total = defaultdict(int)
    per_task_correct = defaultdict(int)

    with torch.no_grad():
        for example in dataset:
            if "prompt" not in example or "completion" not in example:
                raise KeyError("Dataset example must contain 'prompt' and 'completion' fields.")

            if "task_base" not in example:
                raise KeyError("Dataset example must contain 'task_base' for per-task evaluation.")

            prompt = example["prompt"]
            gold = example["completion"]  # "A" or "B"
            task_base = example["task_base"]

            if gold not in ("A", "B"):
                raise ValueError(f"Gold label must be 'A' or 'B', got: {gold}")

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_length,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred = _extract_choice_from_text(gen_text)

            total += 1
            per_task_total[task_base] += 1

            if pred == gold:
                correct += 1
                per_task_correct[task_base] += 1

    if total == 0:
        raise RuntimeError("No examples evaluated; check dataset configuration.")

    overall_acc = correct / total
    per_task_acc = {
        task: per_task_correct[task] / per_task_total[task]
        for task in per_task_total
    }
    return overall_acc, per_task_acc


def main() -> None:
    _validate_config()

    device = torch.device(DEVICE)

    dataset_cfg = SFTDatasetConfig(
        data_root=DATA_ROOT,
        tasks=list(TASKS),
    )

    print("Loading test dataset...")
    test_dataset = load_sft_test_dataset(dataset_cfg)
    print(f"Test size: {len(test_dataset)}")

    print(f"Loading model and tokenizer from {MODEL_NAME_OR_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must have an eos_token defined.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)

    overall_acc, per_task_acc = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        device=device,
    )

    print("\n=== Evaluation Results ===")
    print(f"Overall accuracy: {overall_acc:.4f}")
    for task in sorted(per_task_acc.keys()):
        print(f"Task {task}: accuracy = {per_task_acc[task]:.4f}")


if __name__ == "__main__":
    main()
