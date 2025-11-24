# prepare_splits_shuffle.py
# Use Synthetic data/raw/synthetic_en to generate dedicated folders with diffrent pairs of morel dilemma 
# all dilemma a binary
# "A": "Autonomy",
# "B": "Honesty",
# "C": "Justice"
# script inverts from A>B, B>C, C>A to A<B, B<C, C<A
# script also shuffels ansewrs options so ca. 50% preffered option is either A or B 
# Value mapping adjusted 

import os
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

import random
from pathlib import Path
from src.utils.io import load_jsonl, save_jsonl

SEED = 42
random.seed(SEED)

INPUT_DIR = Path("data/raw/synthetic_en")
OUTPUT_DIR = Path("data/processed/sft_taskvectors")

# -------------------------------
# ORDERED TASKS CONFIG
# -------------------------------
ORDERED_TASKS = {
    "AB": ["AB", "BA"],
    "BC": ["BC", "CB"],
    #"CA": ["CA", "AC"]
}

VALUES = {
    "A": "Autonomy",
    "B": "Honesty",
    "C": "Justice"
}

# -------------------------------
# SHUFFLING (from previous script)
# -------------------------------
def shuffle_answer_side(item):
    """
    Randomly forces the correct answer to be on A or B with 50/50 probability.
    """
    options = item["options"]
    value_mapping = item["value_mapping"]

    # Detect original correct option
    original_correct_key = None
    for k, v in value_mapping.items():
        if v is not None and v != "":
            original_correct_key = k
            break

    if original_correct_key is None:
        return item  # skip

    other_key = "B" if original_correct_key == "A" else "A"
    assign_correct_to_A = random.random() < 0.5

    if assign_correct_to_A:
        new_options = {
            "A": options[original_correct_key],
            "B": options[other_key]
        }
        new_value_mapping = {
            "A": value_mapping[original_correct_key],
            "B": value_mapping[other_key]
        }
    else:
        new_options = {
            "A": options[other_key],
            "B": options[original_correct_key]
        }
        new_value_mapping = {
            "A": value_mapping[other_key],
            "B": value_mapping[original_correct_key]
        }

    item["options"] = new_options
    item["value_mapping"] = new_value_mapping
    return item

# -------------------------------
# SPLITTING
# -------------------------------
def split_data(items, train_size=3200, dev_size=400, test_size=400):
    assert len(items) >= train_size + dev_size + test_size, \
        f"Not enough samples! Only {len(items)} found."

    random.shuffle(items)

    train = items[:train_size]
    dev = items[train_size:train_size+dev_size]
    test = items[train_size+dev_size:train_size+dev_size+test_size]

    return train, dev, test

# -------------------------------
# ORDERED TASKS LABEL DERIVATION
# -------------------------------
def derive_label(item, ordered_task):
    """
    Determines which option letter corresponds to the preferred value.
    Example: ordered_task="BA" → preferred_letter="B" → preferred_value="Honesty"
    """
    preferred_letter = ordered_task[0]
    target_value = VALUES[preferred_letter]

    for opt_letter, principle_name in item["value_mapping"].items():
        if principle_name == target_value:
            return opt_letter

    return None  # skip mismatch

def create_ordered_task_files(base_pair, split_name, items):
    """
    Create derived ordered-task splits from the base-pair items.
    Saves them under: OUTPUT_DIR/tasks/<ordered_task>/<split>.jsonl
    """
    for ordered_task in ORDERED_TASKS[base_pair]:
        out_items = []
        skipped = 0

        for d in items:
            correct_opt = derive_label(d, ordered_task)
            if correct_opt is None:
                skipped += 1
                continue

            d_new = d.copy()
            d_new["task"] = ordered_task
            d_new["correct_option"] = correct_opt
            out_items.append(d_new)

        out_dir = OUTPUT_DIR / "tasks" / ordered_task
        out_dir.mkdir(parents=True, exist_ok=True)
        save_jsonl(out_items, out_dir / f"{split_name}.jsonl")

        print(f"[Ordered Task] {ordered_task}/{split_name}: "
              f"{len(out_items)} items (skipped {skipped})")

# -------------------------------
# MAIN PER-PAIR PROCESSING
# -------------------------------
def process_pair(pair):
    print(f"Processing pair: {pair}")

    raw_file = INPUT_DIR / f"{pair}.jsonl"
    items = load_jsonl(raw_file)
    print(f"Loaded {len(items)} samples.")

    # 1) Shuffle answer-side uniformly
    items = [shuffle_answer_side(item) for item in items]

    # 2) Split into train/dev/test
    train, dev, test = split_data(items)

    # 3) Save base-pair splits
    pair_dir = OUTPUT_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, pair_dir / "train.jsonl")
    save_jsonl(dev, pair_dir / "dev.jsonl")
    save_jsonl(test, pair_dir / "test.jsonl")

    print(f"Saved base-pair splits to {pair_dir}")

    # 4) Also generate ordered-task variants
    create_ordered_task_files(pair, "train", train)
    create_ordered_task_files(pair, "dev", dev)
    create_ordered_task_files(pair, "test", test)

# -------------------------------
# MAIN
# -------------------------------
def main():
    # Now all 3 canonical pairs including CA
    for pair in ["AB", "BC"]: #, "CA"
        process_pair(pair)

if __name__ == "__main__":
    main()
