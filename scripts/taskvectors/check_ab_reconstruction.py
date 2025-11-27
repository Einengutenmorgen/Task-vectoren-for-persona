import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# 1. Setup Path & Cuda (Keep this as is)
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.calculate_taskvec import (
    load_state_dict,
    extract_lora_delta,
)
base_path = "META-LLAMA/LLAMA-3.2-1B"
path_AB   = PROJECT_ROOT / "models/flipped_model_ba_4/AB/merged"
dev_AB    = PROJECT_ROOT / "data/sft_taskvectors/AB/dev.jsonl"


def load_jsonl(p):
    samples = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def format_prompt(sample):
    story = sample["story"].strip()
    question = sample["question"].strip()
    opts = sample["options"]
    opt_a = opts["A"].strip()
    opt_b = opts["B"].strip()
    return (
        "Here is a situation that needs to be analysed. The story:\n\n"
        f"{story}\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n\n"
        'Answer only as "A" or "B". \n Answer:'
    )


def parse_choice(text: str):
    clean = text.strip().upper()
    return clean[0] if clean and clean[0] in ("A", "B") else None


def evaluate(model, tokenizer, dev_samples, device):
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0
    for s in dev_samples:
        prompt = format_prompt(s)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

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

    return correct / total if total else 0.0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dev_samples = load_jsonl(dev_AB)
    print(f"Loaded {len(dev_samples)} AB dev samples.")

    # 1) Evaluate the original AB merged model on AB/dev
    tok_ab = AutoTokenizer.from_pretrained(str(path_AB))
    model_ab = AutoModelForCausalLM.from_pretrained(str(path_AB))
    model_ab.to(device)
    acc_ab = evaluate(model_ab, tok_ab, dev_samples, device)
    print(f"[Original AB] accuracy on AB/dev: {acc_ab:.4f}")

    # 2) Build Δ_AB and reconstruct θ0 + Δ_AB
    theta_0 = load_state_dict(base_path)
    theta_AB = load_state_dict(str(path_AB))
    delta_ab = extract_lora_delta(theta_0, theta_AB)

    # Build new state dict: θ0 + Δ_AB
    recon_sd = {k: v.clone() for k, v in theta_0.items()}
    for k, dv in delta_ab.items():
        recon_sd[k] = theta_0[k] + dv

    # 3) Load reconstructed model and evaluate on AB/dev
    tok_base = AutoTokenizer.from_pretrained(base_path)
    model_recon = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.float32,
        device_map={"": device},
    )
    model_recon.load_state_dict(recon_sd, strict=True)
    acc_recon = evaluate(model_recon, tok_base, dev_samples, device)
    print(f"[Reconstructed base+Δ_AB] accuracy on AB/dev: {acc_recon:.4f}")


if __name__ == "__main__":
    main()
