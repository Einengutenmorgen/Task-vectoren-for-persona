#derive_ordered_tasks.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pathlib import Path
from src.utils.io import load_jsonl, save_jsonl

# mapping base-pair to ordered tasks
ORDERED_TASKS = {
    "AB": ["AB", "BA"],
    "BC": ["BC", "CB"],
    "CA": ["CA", "AC"]
}

VALUES = {
    "A": "Autonomy",
    "B": "Honesty",
    "C": "Justice"
}

BASE_DIR = Path("data/processed/en/base_pairs")
OUTPUT_DIR = Path("data/processed/en/tasks")

def derive_label(item, ordered_task):
    # ordered_task like "AB": prefer A over B
    preferred_letter = ordered_task[0]  # e.g. "C"
    
    target_value = VALUES[preferred_letter] # e.g. "Justice"

    # Determine which option corresponds to preferred value
    for opt_letter, principle_name in item["value_mapping"].items():
        if principle_name == target_value:
            return opt_letter
    
    # CHANGE: Return None instead of raising Error immediately, 
    # so we can handle it gracefully in the loop
    return None 

def process_one_pair(pair):
    for split in ["train", "dev", "test"]:
        file_path = BASE_DIR / pair / f"{split}.jsonl"
        if not file_path.exists():
            print(f"Skipping {file_path}, file not found.")
            continue
            
        items = load_jsonl(file_path)
        
        for ordered_task in ORDERED_TASKS[pair]:
            out_items = []
            skipped_count = 0
            
            for d in items:
                correct_opt = derive_label(d, ordered_task)
                
                # CHANGE: Check if valid label was found
                if correct_opt is None:
                    skipped_count += 1
                    continue

                d_new = d.copy()
                d_new["task"] = ordered_task
                d_new["correct_option"] = correct_opt
                out_items.append(d_new)

            out_dir = OUTPUT_DIR / ordered_task
            out_dir.mkdir(parents=True, exist_ok=True)
            save_jsonl(out_items, out_dir / f"{split}.jsonl")

            print(f"Saved {ordered_task}/{split}: {len(out_items)} items. (Skipped {skipped_count} mismatched items)")

def main():
    for pair in ["AB", "BC", "CA"]:
        process_one_pair(pair)

if __name__ == "__main__":
    main()
