from src.utils.io import load_jsonl
from src.schema.validator import validate_dilemma
from pathlib import Path

def main():
    root = Path("data/raw/synthetic_en")
    for pair in ["AB", "BC", "CA"]:
        data = load_jsonl(root / f"{pair}.jsonl")

        errors = []
        for i, d in enumerate(data):
            ok = validate_dilemma(d)
            if ok is not True:
                errors.append((i, ok))

        print(pair, "valid:", len(data)-len(errors), "invalid:", len(errors))

if __name__ == "__main__":
    main()
