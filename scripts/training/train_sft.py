import os
import sys
from pathlib import Path

# --- Project Root Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

from src.training.sft_dataset import SFTDilemmaDataset

# ==========================================
# GLOBAL CONFIGURATION SECTION
# ==========================================
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = "models/sft"
TRAIN_FILE_ROOT = "data/processed/en/tasks" # Tasks are subdirectories here
BATCH_SIZE = 8
EPOCHS = 5
LR = 2e-4
MAX_SEQ_LEN = 512
# ==========================================

def find_available_tasks(root_dir: str) -> list[str]:
    """Dynamically finds all task directories that contain a train.jsonl file."""
    root = Path(root_dir)
    if not root.exists():
        print(f"Error: Training root directory not found at {root_dir}")
        return []
        
    tasks = []
    # Find all subdirectories that contain 'train.jsonl'
    for item in root.iterdir():
        if item.is_dir() and (item / "train.jsonl").exists():
            tasks.append(item.name)
    return tasks

def run_training_for_task(task: str, device: str):
    """Executes the SFT training process for a single specified task."""
    print(f"\n--- Starting Training for Task: {task} ---")
    
    train_path = Path(TRAIN_FILE_ROOT) / task / "train.jsonl"
    dev_path   = Path(TRAIN_FILE_ROOT) / task / "dev.jsonl"
    
    if not train_path.exists():
        print(f"Skipping task {task}: train file not found at {train_path}")
        return

    try:
        train_ds = SFTDilemmaDataset(train_path, task)
        dev_ds   = SFTDilemmaDataset(dev_path, task) # Dev file is usually optional, but assumed available

        # Convert to HF Dataset for SFTTrainer
        train_hf = Dataset.from_list(list(train_ds))
        dev_hf   = Dataset.from_list(list(dev_ds))
    except Exception as e:
        print(f"Error loading datasets for task {task}: {e}")
        return

    # --- Tokenizer and Model Loading ---
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )

    # --- LoRA Configuration ---
    peft_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "k_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    print(f"Trainable parameters for {task}:")
    model.print_trainable_parameters()

    def formatting_func(examples):
        return examples["text"]

    # Define final output path based on global config and current task
    final_output_dir = Path(OUTPUT_DIR) / "llama-3.2-1B" / "en" / task
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # --- Trainer Setup ---
    training_args = TrainingArguments(
        output_dir=str(final_output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=(device == "cuda"),
        use_mps_device=(device == "mps"),
        load_best_model_at_end=True, # Optional: load best checkpoint
        metric_for_best_model="eval_loss",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_hf,
        eval_dataset=dev_hf,
        peft_config=None,
        args=training_args,
        formatting_func=formatting_func,

    )

    # --- Training Execution ---
    trainer.train()
    
    # Save the final model and tokenizer
    trainer.save_model(str(final_output_dir))
    tokenizer.save_pretrained(str(final_output_dir))
    print(f"Successfully saved LoRA SFT model for task {task} to {final_output_dir}")
    print(f"--- Finished Training for Task: {task} ---")


def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tasks_to_run = find_available_tasks(TRAIN_FILE_ROOT)

    if not tasks_to_run:
        print(f"No tasks found in {TRAIN_FILE_ROOT}. Exiting.")
        return

    print(f"Found {len(tasks_to_run)} tasks to train: {', '.join(tasks_to_run)}")
    
    for task in tasks_to_run:
        run_training_for_task(task, device)

    print("\n\n*** All training tasks completed. ***")

if __name__ == "__main__":
    main()