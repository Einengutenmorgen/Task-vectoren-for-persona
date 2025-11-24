import json
import os
from pathlib import Path
from collections import defaultdict

# Config - Adjust to match your data path
# Based on your previous script: data/processed/en/tasks
DATA_ROOT = Path("data/processed/en/abliation_studies_sft") 
TASKS = ["AB", "BC", "CB"]

def check_file_balance(file_path):
    stats = defaultdict(int)
    total = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Check standard fields for the 'correct' answer
                # Adjust keys 'completion', 'answer', 'gold' based on your schema
                label = data.get('correct_option') or data.get('answer') or data.get('gold')
                
                if label:
                    stats[label.strip().upper()] += 1
                    total += 1
    except FileNotFoundError:
        print(f"!! File not found: {file_path}")
        return None

    return stats, total

def main():
    print(f"Scanning data in: {DATA_ROOT.resolve()}")
    print("-" * 40)

    overall_skewed = False

    for task in TASKS:
        task_dir = DATA_ROOT / task
        # Check for common file names like 'train.jsonl', 'test.jsonl', 'dev.jsonl'
        files = list(task_dir.glob("*.jsonl"))
        
        if not files:
            print(f"Task {task}: No JSONL files found in {task_dir}")
            continue

        print(f"Task: {task}")
        for file_path in files:
            stats, total = check_file_balance(file_path)
            if total == 0:
                print(f"  - {file_path.name}: Empty or no labels found.")
                continue

            a_count = stats.get('A', 0)
            b_count = stats.get('B', 0)
            a_pct = (a_count / total) * 100
            b_pct = (b_count / total) * 100

            print(f"  - {file_path.name}: Total {total}")
            print(f"    A: {a_count} ({a_pct:.1f}%)")
            print(f"    B: {b_count} ({b_pct:.1f}%)")
            
            # Flag if highly imbalanced (e.g., > 90% one side)
            if a_pct > 90 or b_pct > 90:
                print("    [!] WARNING: Highly Imbalanced! Model might learn position bias.")
                overall_skewed = True
        print("-" * 40)

    if overall_skewed:
        print("\nCONCLUSION: Your data is heavily biased towards one option.")
        print("The model likely learned 'Always pick A' rather than the ethical concept.")
        print("You must randomize the option order in your SFT Dataset generation.")
    else:
        print("\nCONCLUSION: Data looks balanced. The issue lies elsewhere (likely prompt formatting).")

if __name__ == "__main__":
    main()