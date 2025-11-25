import json
import torch
import gc
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================
# Part 1: Vector Arithmetic
# =========================

def calculate_task_vectors(path_base, path_AB, path_BC, path_CB, debug=False):
    """
    Compute:
      - theta_0      : base model weights
      - delta_instr  : instruction direction (avg of BC, CB)
      - delta_pref   : preference-only direction (AB with instruction projected out)
      - delta_ab     : raw AB direction (for debugging / reconstruction)
      - eval_model   : a base model instance to be reused for loading new state_dicts
    All math is done in float32 on CPU for stability.
    """
    print(f"--- Starting Vector Extraction (Math on CPU, float32) ---")

    # 1. Load Base Model (theta_0)
    print(f"Loading Base Model: {path_base}")
    base_model = AutoModelForCausalLM.from_pretrained(
        path_base,
        dtype=torch.float32,
        device_map="cpu"
    )
    theta_0 = {k: v.clone() for k, v in base_model.state_dict().items()}
    eval_model_structure = base_model  # we'll reuse this skeleton

    # Optional quick sanity check on base with a toy prompt
    if debug:
        print("\n[DEBUG] Base model toy-prompt sanity check")
        tok_base = AutoTokenizer.from_pretrained(path_base)
        if tok_base.pad_token is None:
            tok_base.pad_token = tok_base.eos_token

        test_prompt = "Say just A or B. Answer:"
        inputs = tok_base(test_prompt, return_tensors="pt").to("cpu")

        with torch.no_grad():
            out = base_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok_base.pad_token_id
            )

        print("[BASE TEST] decoded:",
              tok_base.decode(out[0], skip_special_tokens=False))

        print("\n[DEBUG] AB merged toy-prompt sanity check")
        tok_ab = AutoTokenizer.from_pretrained(path_AB)
        if tok_ab.pad_token is None:
            tok_ab.pad_token = tok_ab.eos_token
        model_ab_test = AutoModelForCausalLM.from_pretrained(
            path_AB,
            dtype=torch.float32,
            device_map="cpu"
        )

        inputs = tok_ab(test_prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            out_ab = model_ab_test.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok_ab.pad_token_id
            )

        print("[AB TEST] decoded:",
              tok_ab.decode(out_ab[0], skip_special_tokens=False))

    # 2. Calculate Instruction Vector from BC & CB
    print("Computing Helper Deltas (BC & CB)...")

    print("Loading BC...")
    model_bc = AutoModelForCausalLM.from_pretrained(
        path_BC,
        dtype=torch.float32,
        device_map="cpu"
    )
    theta_bc = model_bc.state_dict()
    delta_bc = {k: theta_bc[k] - theta_0[k] for k in theta_bc}
    del model_bc, theta_bc
    gc.collect()

    print("Loading CB...")
    model_cb = AutoModelForCausalLM.from_pretrained(
        path_CB,
        dtype=torch.float32,
        device_map="cpu"
    )
    theta_cb = model_cb.state_dict()
    delta_cb = {k: theta_cb[k] - theta_0[k] for k in theta_cb}
    del model_cb, theta_cb
    gc.collect()

    print("Creating Delta_instr...")
    delta_instr = {}
    for k in delta_bc:
        delta_instr[k] = 0.5 * (delta_bc[k] + delta_cb[k])
    del delta_bc, delta_cb
    gc.collect()

    # 3. Calculate Preference Vector (AB)
    print("Computing Source Delta (AB)...")
    model_ab = AutoModelForCausalLM.from_pretrained(
        path_AB,
        dtype=torch.float32,
        device_map="cpu"
    )
    theta_ab = model_ab.state_dict()
    delta_ab = {k: theta_ab[k] - theta_0[k] for k in theta_ab}
    del model_ab, theta_ab
    gc.collect()

    # 4. Orthogonal Projection to remove instruction component
    print("Calculating Projection Alpha...")
    dot_prod = 0.0
    norm_sq = 0.0

    for k in delta_instr:
        flat_ab = delta_ab[k].float().view(-1)
        flat_instr = delta_instr[k].float().view(-1)
        dot_prod += torch.dot(flat_ab, flat_instr)
        norm_sq += torch.dot(flat_instr, flat_instr)

    alpha = dot_prod / norm_sq
    print(f"Calculated Alpha: {alpha.item():.6f}")

    print("Creating Delta_pref_only...")
    delta_pref_only = {}
    avg_norm = 0.0
    count = 0
    for k in delta_ab:
        vec = delta_ab[k] - (alpha * delta_instr[k])
        delta_pref_only[k] = vec

        # DEBUG: Check norm of first few layers
        if count < 5:
            n = torch.norm(vec.float()).item()
            avg_norm += n
            count += 1

    if count > 0:
        print(f"DEBUG: Average norm of first {count} layers of Pref Vector: {avg_norm/count:.4f}")
    if avg_norm == 0:
        print("!!! WARNING: Preference Vector appears to be all zeros! Check inputs.")

    return eval_model_structure, theta_0, delta_instr, delta_pref_only, delta_ab


# =========================
# Part 2: Model Composition
# =========================

def build_flipped_state_dict(theta_0, delta_instr, delta_pref, gamma1, gamma2):
    """
    theta_new = theta_0 + gamma1 * delta_instr - gamma2 * delta_pref
    """
    theta_new = {}
    with torch.no_grad():
        for k in theta_0:
            theta_new[k] = (
                theta_0[k]
                + (gamma1 * delta_instr[k])
                - (gamma2 * delta_pref[k])
            )
    return theta_new


# =========================
# Part 3: Data Loading
# =========================

def load_dev_set(jsonl_path):
    samples = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except Exception:
                    pass
    except FileNotFoundError:
        print(f"ERROR: File not found at {jsonl_path}")
        return []
    return samples


def format_prompt(sample):
    story = sample["story"].strip()
    question = sample["question"].strip()
    opt_a = sample.get("option_A", "Option A").strip()
    opt_b = sample.get("option_B", "Option B").strip()

    prompt = (
        "Here is a situation that needs to be analysed. The story:\n\n"
        f"{story}\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A. {opt_a}\n"
        f"B. {opt_b}\n\n"
        'Answer only as "A" or "B". \n Answer:'
    )
    return prompt


def parse_choice_from_output(text: str):
    clean = text.strip().upper()
    if len(clean) == 0:
        return None
    first_char = clean[0]
    if first_char in ("A", "B"):
        return first_char
    return None


# =========================
# Part 4: Evaluation
# =========================

def evaluate_model(model, tokenizer, dev_samples, device, debug_print=False):
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0

    for i, sample in enumerate(dev_samples):
        prompt = format_prompt(sample)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

        if debug_print and i < 3:
            print(f"[DEBUG ids]: {generated_ids}")
            print("[DEBUG raw]:",
                  tokenizer.decode(generated_ids, skip_special_tokens=False))

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        choice = parse_choice_from_output(generated_text)

        gold = sample["correct_option"].strip().upper()

        if debug_print and i < 3:
            print(f"[DEBUG decoded]: {generated_text!r} -> Parsed: {choice}, Gold: {gold}")

        total += 1
        if choice == gold:
            correct += 1

    return correct / total if total > 0 else 0.0


def grid_search(eval_model, tokenizer, theta_0, delta_instr, delta_pref, device, dev_samples):
    g1_range = [i * 0.3 for i in range(0, 11)]  # 0.0 ... 3.0
    g2_range = [i * 0.3 for i in range(0, 8)]   # 0.0 ... 2.1

    best_acc = -1.0
    best_gammas = (0.0, 0.0)

    print(f"Starting coarse grid search over {len(g1_range) * len(g2_range)} combinations...")
    eval_model.to(device)

    for g1 in g1_range:
        for g2 in g2_range:
            print(f"Testing: g1={g1:.2f}, g2={g2:.2f} ...", end=" ")

            new_state_dict = build_flipped_state_dict(theta_0, delta_instr, delta_pref, g1, g2)

            try:
                eval_model.load_state_dict(new_state_dict, strict=True)
            except RuntimeError as e:
                print("\nCRITICAL ERROR: Key mismatch in load_state_dict. "
                      "The vectors don't match the model structure.")
                print(e)
                sys.exit(1)

            debug = (g1 == 0.0 and g2 == 0.0)  # debug first config only

            acc = evaluate_model(eval_model, tokenizer, dev_samples, device, debug_print=debug)
            print(f"Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                best_gammas = (g1, g2)

    print(f"Best Accuracy: {best_acc:.4f} | Optimal: g1={best_gammas[0]:.2f}, g2={best_gammas[1]:.2f}")
    return best_gammas


# =========================
# Part 5: Main
# =========================

if __name__ == "__main__":
    # Paths
    base_path = "META-LLAMA/LLAMA-3.2-1B"
    path_AB = "models/sft_tv/AB/merged"
    path_BC = "models/sft_tv/BC/merged"
    path_CB = "models/sft_tv/CB/merged"
    dev_path_BA = "data/sft_taskvectors/BA/dev.jsonl"  # BA dev for flipping AB->BA

    DEBUG = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Inference Device: {device}")

    # Calculate vectors (on CPU)
    eval_model, theta_0, delta_instr, delta_pref, delta_ab = calculate_task_vectors(
        base_path, path_AB, path_BC, path_CB, debug=DEBUG
    )

    # Move model to device if needed
    eval_model = eval_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dev_samples_BA = load_dev_set(dev_path_BA)
    if not dev_samples_BA:
        print("No dev samples loaded; exiting.")
        sys.exit(1)

    if DEBUG:
        # AB merged on real dev prompt
        print("\n[DEBUG] AB MERGED on REAL BA dev prompt")
        tok_ab = AutoTokenizer.from_pretrained(path_AB)
        if tok_ab.pad_token is None:
            tok_ab.pad_token = tok_ab.eos_token
        model_ab_test = AutoModelForCausalLM.from_pretrained(
            path_AB,
            dtype=torch.float32,
            device_map="cpu"
        )
        model_ab_test.eval()

        sample = dev_samples_BA[0]
        prompt = format_prompt(sample)
        inputs = tok_ab(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cpu")

        with torch.no_grad():
            out = model_ab_test.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tok_ab.pad_token_id
            )

        tail_ids = out[0][inputs["input_ids"].shape[1]:]
        print("AB tail ids:", tail_ids)
        print("AB decoded (no skip):", tok_ab.decode(tail_ids, skip_special_tokens=False))
        print("AB decoded (skip):    ", tok_ab.decode(tail_ids, skip_special_tokens=True))

        # Base model on real dev prompt (theta_0)
        print("\n[DEBUG] Base model (theta_0) on REAL BA dev prompt")
        sample = dev_samples_BA[0]
        prompt = format_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)

        with torch.no_grad():
            out = eval_model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        tail_ids = out[0][inputs["input_ids"].shape[1]:]
        print("Base tail ids:", tail_ids)
        print("Base decoded (no skip):", tokenizer.decode(tail_ids, skip_special_tokens=False))
        print("Base decoded (skip):    ", tokenizer.decode(tail_ids, skip_special_tokens=True))

    # Grid search over gammas using BA dev set
    best_g1, best_g2 = grid_search(
        eval_model, tokenizer, theta_0, delta_instr, delta_pref, device, dev_samples_BA
    )

    print(f"\nSaving best flipped model (AB -> BA)...")
    final_state_dict = build_flipped_state_dict(theta_0, delta_instr, delta_pref, best_g1, best_g2)
    eval_model.load_state_dict(final_state_dict)
    eval_model.save_pretrained(".models/flipped_model_ba_4")
    tokenizer.save_pretrained(".models/flipped_model_ba_4")
    print("Done.")
