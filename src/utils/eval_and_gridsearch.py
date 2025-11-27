import os
import json
import torch
import gc
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import corrected taskvec utilities
from src.utils.calculate_taskvec import (
    calculate_task_vectors_fixed,
    build_flipped_model,
)

# ======================================================
# 1. Load JSONL dev set
# ======================================================
def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except:
                pass
    return samples


# ======================================================
# 2. Prompt formatting – identical to training script
# ======================================================
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


# ======================================================
# 3. Extract A/B from model-generated text
# ======================================================
def parse_choice(text: str):
    clean = text.strip().upper()
    if not clean:
        return None
    if clean[0] in ("A", "B"):
        return clean[0]
    return None


# ======================================================
# 4. Evaluate a model on a dev set
# ======================================================
def evaluate_model(model, tokenizer, dev_samples, device="cuda"):
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


# ======================================================
# 5. γ-grid search (uses corrected Δ_instr + Δ_pref)
# ======================================================
def gamma_grid_search(
    base_path,
    theta_0,
    delta_instr,
    delta_pref,
    dev_samples,
    save_tmp_dir,
    device="cuda"
):
    print("\n=== Starting γ-grid search ===")

    # Paper-guided ranges: #list comprehension returns 0.1 increments 
    # g1 = 1
    # g1_range = [i / 10 for i in range(int(g1 * 10) + 1)] 
    # g2= 1
    # g2_range = [i / 10 for i in range(int(g2 * 10) + 1)]
    g1_range=[0.7, 0.8, 0.9]
    g2_range=[0.2, 0.3 , 0.4]

    best_acc = -1.0
    best_g1 = None
    best_g2 = None

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tmp_path = Path(save_tmp_dir)
    tmp_path.mkdir(parents=True, exist_ok=True)

    for g1 in g1_range:
        for g2 in g2_range:
            print(f"Testing γ1={g1:.2f}, γ2={g2:.2f} ... ", end="")

            # Build temporary checkpoint
            candidate_dir = tmp_path / f"tmp_g1_{g1}_g2_{g2}"
            if candidate_dir.exists():
                # avoid stale content
                for f in candidate_dir.iterdir():
                    f.unlink()

            model = build_flipped_model(
                base_path=base_path,
                delta_instr=delta_instr,
                delta_pref=delta_pref,
                gamma1=g1,
                gamma2=g2,
                save_dir=str(candidate_dir),
                do_save=False
            )

            # Evaluate
            acc = evaluate_model(model, tokenizer, dev_samples, device=device)
            print(f"Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_g1 = g1
                best_g2 = g2

            # cleanup
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n=== GRID SEARCH DONE ===")
    print(f"Best accuracy = {best_acc:.4f}")
    print(f"Best gammas: γ1 = {best_g1}, γ2 = {best_g2}")

    return best_g1, best_g2, best_acc


# ======================================================
# 6. High-level DRIVER
# ======================================================
def run_task_vector_alignment(
    base_path,
    path_AB,
    path_BC,
    path_CB,
    dev_jsonl,
    save_tmp_dir="tmp_taskvec",
    save_final_dir="model_flipped_final",
    device="cuda"
):
    print("\n\n=== Loading dev set ===")
    dev_samples = load_jsonl(dev_jsonl)
    print(f"Loaded {len(dev_samples)} dev samples.\n")

    # ------------------------------------------------------
    # Compute corrected Δ vectors
    # ------------------------------------------------------
    theta_0, delta_instr, delta_pref = calculate_task_vectors_fixed(
        base_path,
        path_AB,
        path_BC,
        path_CB,
        debug=True
    )

    # ------------------------------------------------------
    # γ-grid search
    # ------------------------------------------------------
    best_g1, best_g2, best_acc = gamma_grid_search(
        base_path=base_path,
        theta_0=theta_0,
        delta_instr=delta_instr,
        delta_pref=delta_pref,
        dev_samples=dev_samples,
        save_tmp_dir=save_tmp_dir,
        device=device
    )

    # ------------------------------------------------------
    # Build final flipped model
    # ------------------------------------------------------
    print("\n=== BUILDING FINAL FLIPPED MODEL ===")
    build_flipped_model(
        base_path=base_path,
        delta_instr=delta_instr,
        delta_pref=delta_pref,
        gamma1=best_g1,
        gamma2=best_g2,
        save_dir=save_final_dir,
        do_save=True
    )

    print("\n=== DONE ===\n")
    return best_g1, best_g2, best_acc
