import json

def load_jsonl(path):
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(items, path):
    with open(path, "w", encoding="utf8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
