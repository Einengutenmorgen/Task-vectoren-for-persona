from __future__ import annotations

import os 
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer



# Set up Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.training.sft_dataset import SFTDatasetConfig, load_sft_test_dataset

# =========================
# EXPLICIT CONFIGURATION
# =========================


MODEL_NAME_OR_PATH: str | None = 'data/results/en/AB/merged' 
DATA_ROOT: str | None = str(PROJECT_ROOT / 'data/processed/en/tasks')

TASKS = ("AB", "BC", "CA")

MAX_SEQ_LENGTH: int = 512
MAX_NEW_TOKENS: int = 4 # Increased slightly to catch "Answer: A" formats if they occur
DEVICE: str = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


def _validate_config() -> None:
    if not MODEL_NAME_OR_PATH:
        raise ValueError("MODEL_NAME_OR_PATH must be set.")
    if not DATA_ROOT:
        raise ValueError("DATA_ROOT must be set.")
    print(f"Configuration Validated. Device: {DEVICE}")


def _extract_choice_from_text(text: str) -> Optional[str]:
    """
    Extract the first 'A' or 'B' from generated text.
    Returns None if not found (instead of crashing).
    """
    # Upper case to handle 'a' or 'b'
    for ch in text.upper():
        if ch in ("A", "B"):
            return ch
    return None


def evaluate_model(
    model,
    tokenizer,
    dataset,
    max_seq_length: int,
    max_new_tokens: int,
    device: str,
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

    print(f"Starting evaluation on {len(dataset)} examples...")

    with torch.no_grad():
        for i, example in enumerate(dataset):
            prompt = example.get("prompt")
            gold = example.get("completion")
            task_base = example.get("task_base")

            if not prompt or not gold or not task_base:
                print(f"Skipping invalid example {i}")
                continue

            # Ensure inputs are on the correct device
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
                pad_token_id=tokenizer.pad_token_id, # Explicitly use the tokenizer's pad token
            )[0]

            input_len = inputs["input_ids"].shape[1]
            gen_ids = output_ids[input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            pred = _extract_choice_from_text(gen_text)

            # Debug print for the first few examples
            if i < 3:
                print(f"[Example {i}] Task: {task_base} | Gold: {gold} | Pred: {pred} | Raw Output: {repr(gen_text)}")

            total += 1
            per_task_total[task_base] += 1

            if pred == gold:
                correct += 1
                per_task_correct[task_base] += 1
            
            # Optional: Print progress every 50 examples
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(dataset)}...")

    if total == 0:
        print("Warning: No valid examples evaluated.")
        return 0.0, {}

    overall_acc = correct / total
    per_task_acc = {
        task: per_task_correct[task] / per_task_total[task]
        for task in per_task_total if per_task_total[task] > 0
    }
    return overall_acc, per_task_acc


def main() -> None:
    _validate_config()

    dataset_cfg = SFTDatasetConfig(
        data_root=DATA_ROOT,
        tasks=list(TASKS),
    )

    print("Loading test dataset...")
    test_dataset = load_sft_test_dataset(dataset_cfg)
    
    print(f"Loading tokenizer from {MODEL_NAME_OR_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    
    # CRITICAL FIX: Set padding side to left for generation
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # Fallback if neither exists (unlikely for Llama)
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print(f"Loading model from {MODEL_NAME_OR_PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_OR_PATH)

    overall_acc, per_task_acc = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        dataset=test_dataset,
        max_seq_length=MAX_SEQ_LENGTH,
        max_new_tokens=MAX_NEW_TOKENS,
        device=DEVICE,
    )

    print("\n=== Evaluation Results ===")
    print(f"Model: {MODEL_NAME_OR_PATH}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("-" * 30)
    for task in sorted(per_task_acc.keys()):
        print(f"Task {task}: accuracy = {per_task_acc[task]:.4f}")
    print("-" * 30)


if __name__ == "__main__":
    main()