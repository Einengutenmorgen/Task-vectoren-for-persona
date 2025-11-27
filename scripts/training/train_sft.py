from __future__ import annotations

import os
# 1. Set Device Masking BEFORE other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import sys
import gc
from pathlib import Path

# Set up Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from peft import LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

from src.training.sft_dataset import SFTDatasetConfig, load_sft_train_eval_datasets

# =========================
# EXPLICIT CONFIGURATION
# =========================

MODEL_NAME_OR_PATH: str = "META-LLAMA/LLAMA-3.2-1B" 
DATA_ROOT: str = str(PROJECT_ROOT / 'data/sft_taskvectors') 
BASE_OUTPUT_DIR: str = str(PROJECT_ROOT / 'models/sft_seed') 

TASKS = ("AB", "BC", "CB")

NUM_TRAIN_EPOCHS: float = 5.0
PER_DEVICE_TRAIN_BATCH_SIZE: int = 8
LEARNING_RATE: float = 2e-4
WEIGHT_DECAY: float = 0.01
MAX_LENGTH: int = 512
GRADIENT_ACCUMULATION_STEPS: int = 1

LORA_R: int = 4
LORA_ALPHA: int = 8
SAVE_MERGED: bool = True

LOGGING_STEPS: int = 100
SAVE_TOTAL_LIMIT: int = 2
WARMUP_RATIO: float = 0.03

GLOBAL_SEED = 42

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def _validate_config() -> None:
    """Fail fast if config is invalid or GPU is missing."""
    if not MODEL_NAME_OR_PATH:
        raise ValueError("MODEL_NAME_OR_PATH must be set.")
    if not TASKS:
        raise ValueError("TASKS must contain at least one task.")
    
    # Fail-Fast: Verify CUDA availability immediately
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check your driver or CUDA_VISIBLE_DEVICES setting.")
    
    # Verify we are on the expected logical device (0) which maps to physical (3)
    print(f"Running on: {torch.cuda.get_device_name(0)}")

def check_label_balance_hf(ds, label_key="correct_option", tolerance=0.10):
    """
    Checks whether label distribution in a HuggingFace Dataset is balanced enough.
    
    Args:
        ds: HF Dataset
        label_key: column containing labels
        tolerance: allowed deviation from 50/50

    Behavior:
        - Logs distribution
        - Allows training if deviation <= tolerance
        - Raises RuntimeError to stop training otherwise
    """

    if label_key not in ds.column_names:
        raise ValueError(f"Dataset has no column '{label_key}'")

    # Extract column (fast, memory-efficient)
    labels = ds[label_key]

    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.values())

    print("\n=== Label Distribution Check ===")
    print(f"Total items: {total}")
    for lbl, cnt in counts.items():
        print(f"  {lbl}: {cnt}  ({cnt/total:.2%})")

    # Only apply balance check for a binary dataset (A/B)
    if len(counts) == 2:
        # Convert dict values to sorted order for consistency
        labels_list = sorted(list(counts.keys()))
        p = counts[labels_list[0]] / total
        deviation = abs(p - 0.5)

        print(f"Deviation from perfect balance: {deviation:.3f} (tolerance={tolerance:.3f})")

        if deviation > tolerance:
            raise RuntimeError(
                f"[FATAL] Label imbalance too high! "
                f"Deviation={deviation:.3f} exceeds tolerance={tolerance:.3f}"
            )
        else:
            print("[OK] Label balance within tolerance.\n")

    else:
        print("[INFO] Non-binary labels; balance check skipped.\n")


def cleanup_gpu():
    """Cleans up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_single_task(task_name: str) -> None:
    print(f"\n{'='*40}")
    print(f"STARTING TRAINING FOR TASK: {task_name}")
    print(f"{'='*40}\n")

    task_output_dir = os.path.join(BASE_OUTPUT_DIR, task_name)
    os.makedirs(task_output_dir, exist_ok=True)

    # 1. Load Data
    dataset_cfg = SFTDatasetConfig(data_root=DATA_ROOT, tasks=[task_name])
    print(f"[{task_name}] Loading datasets...")
    train_dataset, eval_dataset = load_sft_train_eval_datasets(dataset_cfg)
    print(f"[{task_name}] Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")
    check_label_balance_hf(train_dataset)

    # 2. Load Tokenizer
    print(f"[{task_name}] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    
    # Fix: Handle padding token strictly
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Explicitly set padding side

    # 3. Load Model EXPLICITLY (Fixes 'Device is missing' / Config mismatch)
    print(f"[{task_name}] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map={"": 0}, # Force to local GPU 0 (Physical 3)
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation="sdpa" if torch.cuda.get_device_capability()[0] >= 8 else "eager"
    )

    # Fix: Sync model config with tokenizer changes
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False # Critical: Disable KV cache during training

    # 4. Configure Training
    sft_config = SFTConfig(
        output_dir=task_output_dir,
        run_name=f"sft_{task_name}",
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        max_length=MAX_LENGTH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        lr_scheduler_type="linear",
        warmup_ratio=WARMUP_RATIO,
        bf16=torch.cuda.is_bf16_supported(), # Dynamic check
        tf32=True,
        report_to='wandb',
        packing=False 
    )

    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj"],
    )

    # 5. Initialize Trainer with EXPLICIT model
    print(f"[{task_name}] Initializing Trainer...")
    trainer = SFTTrainer(
        model=model, # Pass the loaded object, not the string
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 6. Train
    print(f"[{task_name}] Training...")
    trainer.train()

    # 7. Save Adapter
    print(f"[{task_name}] Saving adapters...")
    trainer.save_model(task_output_dir)
    tokenizer.save_pretrained(task_output_dir)

    # 8. Merge and Save Full Model (Fixes Wrapper issues)
    if SAVE_MERGED:
        print(f"[{task_name}] Merging weights...")
        
        # Fix: Ensure we are merging the correct object and handling memory
        # We reload or use the trainer model, but we must be careful with 'merge_and_unload' availability
        try:
            # Try merging directly if the method exists on the trainer.model (typical for PEFT)
            if hasattr(trainer.model, "merge_and_unload"):
                merged = trainer.model.merge_and_unload()
            # If wrapped (e.g. by Accelerator), unwrap first
            elif hasattr(trainer.model, "module") and hasattr(trainer.model.module, "merge_and_unload"):
                merged = trainer.model.module.merge_and_unload()
            else:
                # Fallback: PeftModel wrapper might be deeper
                print(f"[{task_name}] Warning: Model wrapped deeply. Attempting manual merge fallback.")
                # For academic fail-fast, we might just skip or try standard unwind
                merged = trainer.model

            merged_dir = os.path.join(task_output_dir, "merged")
            os.makedirs(merged_dir, exist_ok=True)
            
            merged.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            print(f"[{task_name}] Merged model saved to: {merged_dir}")

        except Exception as e:
            print(f"[{task_name}] CRITICAL ERROR DURING MERGE: {e}")
            # Don't stop the pipeline, but log it.
            pass

    # 9. Cleanup
    print(f"[{task_name}] Cleaning up memory...")
    
    # Force deletion of references
    del trainer
    del model
    if 'merged' in locals(): del merged
    
    cleanup_gpu()
    print(f"[{task_name}] Done.\n")


def main() -> None:
    set_global_seed(seed=GLOBAL_SEED)
    _validate_config()
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    print(f"Pipeline started. Training separate models for: {TASKS}")
    
    for task in TASKS:
        try:
            train_single_task(task)
        except Exception as e:
            print(f"CRITICAL ERROR while training task {task}: {e}")
            raise e

    print("All tasks completed successfully.")


if __name__ == "__main__":
    main()