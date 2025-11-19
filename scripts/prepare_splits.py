import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
from pathlib import Path

from src.utils.io import load_jsonl, save_jsonl

SEED = 42
random.seed(SEED)

INPUT_DIR = Path("data/raw/synthetic_en")
OUTPUT_DIR = Path("data/processed/en/base_pairs")

def split_data(items, train_size=3200, dev_size=400, test_size=400):
    assert len(items) >= train_size + dev_size + test_size, \
        f"Not enough samples! Only {len(items)} found."

    random.shuffle(items)

    train = items[:train_size]
    dev = items[train_size:train_size+dev_size]
    test = items[train_size+dev_size:train_size+dev_size+test_size]

    return train, dev, test

def process_pair(pair):
    print(f"Processing pair: {pair}")

    raw_file = INPUT_DIR / f"{pair}.jsonl"
    items = load_jsonl(raw_file)
    print(f"Loaded {len(items)} samples.")

    train, dev, test = split_data(items)

    pair_dir = OUTPUT_DIR / pair
    pair_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train, pair_dir / "train.jsonl")
    save_jsonl(dev, pair_dir / "dev.jsonl")
    save_jsonl(test, pair_dir / "test.jsonl")

    print(f"Saved splits to {pair_dir}")

def main():
    for pair in ["AB", "BC", "CA"]:
        process_pair(pair)

if __name__ == "__main__":
    main()
