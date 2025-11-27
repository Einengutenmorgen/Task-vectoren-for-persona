from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, json
import sys
import os
from pathlib import Path

# 1. Setup Path & Cuda (Keep this as is)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.append(str(project_root))

from src.utils.calculate_taskvec import (
    load_state_dict,
    extract_lora_delta,
)

base_path = "META-LLAMA/LLAMA-3.2-1B"
path_AB = str(project_root / "models/flipped_model_ba_4/AB/merged")
dev_AB_path = str(project_root / "data/sft_taskvectors/AB/dev.jsonl")

# 1. load sd
theta_0 = load_state_dict(base_path)
theta_AB = load_state_dict(path_AB)

# 2. extract delta_AB (LoRA-only)
delta_ab = extract_lora_delta(theta_0, theta_AB)

# --- DEBUG ADDITION ---
print(f"Number of keys in delta: {len(delta_ab)}")
if len(delta_ab) > 0:
    first_key = list(delta_ab.keys())[0]
    print(f"Example Key: {first_key}")
    print(f"Max value in first delta tensor: {delta_ab[first_key].max()}")
    print(f"Sum of first delta tensor: {delta_ab[first_key].sum()}")
else:
    print("CRITICAL: Delta is empty!")

# 3. rebuild theta_0 + delta_ab
recon_sd = {k: v.clone() for k, v in theta_0.items()}
for k in delta_ab:
    recon_sd[k] = theta_0[k] + delta_ab[k]

# 4. load into model
model = AutoModelForCausalLM.from_pretrained(
    base_path,
    dtype=torch.float32,
    device_map={"": 0}
)
model.load_state_dict(recon_sd, strict=True)
model.eval()

# 5. evaluate on AB/dev
tokenizer = AutoTokenizer.from_pretrained(base_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def format_prompt(sample):
    story = sample["story"].strip()
    question = sample["question"].strip()
    opt_a = sample.get("option_A", "").strip()
    opt_b = sample.get("option_B", "").strip()
    return (
        "Here is a situation that needs to be analysed. The story:\n\n"
        f"{story}\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n\n"
        'Answer only as "A" or "B".'
    )

def parse_choice(text: str):
    clean = text.strip().upper()
    return clean[0] if clean and clean[0] in ("A", "B") else None

dev_samples = load_jsonl(dev_AB_path)
correct = 0
total = 0
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

for s in dev_samples:
    prompt = format_prompt(s)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    tail = out[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(tail, skip_special_tokens=True)
    pred = parse_choice(decoded)
    gold = s["correct_option"].strip().upper()
    total += 1
    if pred == gold:
        correct += 1

print(f"Reconstructed AB accuracy: {correct/total:.4f}")
