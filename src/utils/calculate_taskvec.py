import torch
import json
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===========================================================
# 1. HELPER: Restrict keys to LoRA-modified weights
# ===========================================================
LORA_KEYS = ["q_proj", "k_proj", "v_proj"]

def is_lora_key(k):
    return any(sub in k for sub in LORA_KEYS)


# ===========================================================
# 2. Load state dict (CPU, float32)
# ===========================================================
def load_state_dict(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
        device_map="cpu"
    )
    sd = {k: v.clone().float() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    return sd


# ===========================================================
# 3. Extract only LoRA deltas between θ_task and θ_0
# ===========================================================
def extract_lora_delta(theta_0, theta_task):
    delta = {}
    for k in theta_task:
        if is_lora_key(k):
            delta[k] = theta_task[k] - theta_0[k]
    return delta


# ===========================================================
# 4. Average BC & CB instruction deltas
# ===========================================================
def compute_instruction_vector(delta_BC, delta_CB):
    delta_instr = {}
    for k in delta_BC:
        delta_instr[k] = 0.5 * (delta_BC[k] + delta_CB[k])
    return delta_instr


# ===========================================================
# 5. Compute α for projection (LoRA-only layers)
# ===========================================================
def compute_alpha(delta_ab, delta_instr):
    dot_prod = 0.0
    norm_sq = 0.0

    for k in delta_instr:
        v_ab = delta_ab[k].view(-1)
        v_instr = delta_instr[k].view(-1)
        dot_prod += torch.dot(v_ab, v_instr)
        norm_sq += torch.dot(v_instr, v_instr)

    alpha = dot_prod / norm_sq
    return alpha


# ===========================================================
# 6. Remove instructional component → preference-only vector
# ===========================================================
def compute_preference_vector(delta_ab, delta_instr, alpha):
    delta_pref = {}
    for k in delta_ab:
        delta_pref[k] = delta_ab[k] - alpha * delta_instr[k]
    return delta_pref


# ===========================================================
# 7. Build θ_new = θ_0 + γ₁ Δ_instr − γ₂ Δ_pref (LoRA keys only)
# ===========================================================
def build_new_weights(theta_0, delta_instr, delta_pref, gamma1, gamma2):
    new_sd = {k: v.clone() for k, v in theta_0.items()}

    for k in new_sd:
        if is_lora_key(k):
            new_sd[k] = (
                theta_0[k]
                + gamma1 * delta_instr[k]
                - gamma2 * delta_pref[k]
            )

    return new_sd


# ===========================================================
# 8. MAIN ENTRYPOINT
# ===========================================================
def calculate_task_vectors_fixed(
    path_base,
    path_AB,
    path_BC,
    path_CB,
    debug=True
):
    print("=== Corrected Task Vector Extraction ===")

    # -------------------------------------------------------
    # Load base model weights θ₀
    # -------------------------------------------------------
    print("Loading BASE...")
    theta_0 = load_state_dict(path_base)

    # -------------------------------------------------------
    # Load AB model → Δ_AB
    # -------------------------------------------------------
    print("Loading AB...")
    theta_AB = load_state_dict(path_AB)
    delta_ab = extract_lora_delta(theta_0, theta_AB)
    del theta_AB; gc.collect()
    delta_stats("Δ_AB", delta_ab)


    # -------------------------------------------------------
    # Load BC and CB → Δ_BC, Δ_CB
    # -------------------------------------------------------
    print("Loading BC...")
    theta_BC = load_state_dict(path_BC)
    delta_BC = extract_lora_delta(theta_0, theta_BC)
    del theta_BC; gc.collect()
    delta_stats("Δ_BC", delta_BC)


    print("Loading CB...")
    theta_CB = load_state_dict(path_CB)
    delta_CB = extract_lora_delta(theta_0, theta_CB)
    del theta_CB; gc.collect()
    delta_stats("Δ_CB", delta_CB)



    # -------------------------------------------------------
    # Compute instruction vector Δ_instr
    # -------------------------------------------------------
    print("Computing Δ_instr...")
    delta_instr = compute_instruction_vector(delta_BC, delta_CB)
    delta_stats("Δ_instr", delta_instr)


    # -------------------------------------------------------
    # Compute α
    # -------------------------------------------------------
    print("Computing α for projection...")
    alpha = compute_alpha(delta_ab, delta_instr)
    print(f"Alpha = {alpha.item():.6f}")

    # -------------------------------------------------------
    # Compute Δ_pref (preference-only)
    # -------------------------------------------------------
    print("Computing preference-only Δ_pref...")
    delta_pref = compute_preference_vector(delta_ab, delta_instr, alpha)
    delta_stats("Δ_pref", delta_pref)


    # Norm check
    if debug:
        keys = list(delta_pref.keys())[:5]
        print("Sample Δ_pref norms:")
        for k in keys:
            print(f"  {k}: {torch.norm(delta_pref[k]).item():.6f}")
        
    # cosine sanity checks
    cos_ab_instr = cosine_between_deltas(delta_ab, delta_instr)
    cos_ab_pref = cosine_between_deltas(delta_ab, delta_pref)
    cos_instr_pref = cosine_between_deltas(delta_instr, delta_pref)
    print(f"cos(Δ_AB, Δ_instr) = {cos_ab_instr:.4f}")
    print(f"cos(Δ_AB, Δ_pref)  = {cos_ab_pref:.4f}")
    print(f"cos(Δ_instr, Δ_pref) = {cos_instr_pref:.4f}")

    return theta_0, delta_instr, delta_pref


# ===========================================================
# 9. EASY FINAL MODEL BUILDER
# ===========================================================
def build_flipped_model(
    base_path,
    delta_instr,
    delta_pref,
    gamma1,
    gamma2,
    save_dir,
    do_save=True
):
    print(f"Constructing flipped model with γ1={gamma1}, γ2={gamma2}")

    # Load base model structure
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        dtype=torch.float32,
        device_map="cpu"
    )
    sd0 = model.state_dict()

    # Build new SD
    new_sd = build_new_weights(sd0, delta_instr, delta_pref, gamma1, gamma2)

    # Load weights
    model.load_state_dict(new_sd, strict=True)

    if do_save:
        model.save_pretrained(save_dir)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved flipped model to: {save_dir}")
    else:
        print("Skipping save to disk (in-memory only).")
        
    return model

# ===========================================================
# Helper methods to check delta in 
# ===========================================================


def delta_stats(name, delta):
    norms = []
    for v in delta.values():
        norms.append(torch.norm(v).item())
    if not norms:
        print(f"[{name}] has 0 tensors! (BUG)")
        return
    avg = sum(norms) / len(norms)
    mn = min(norms)
    mx = max(norms)
    print(f"[{name}] {len(norms)} tensors | mean={avg:.6f} | min={mn:.6f} | max={mx:.6f}")


def cosine_between_deltas(delta1, delta2):
    num = 0.0
    denom1 = 0.0
    denom2 = 0.0
    for k in delta1:
        if k not in delta2:
            continue
        v1 = delta1[k].view(-1)
        v2 = delta2[k].view(-1)
        num += torch.dot(v1, v2)
        denom1 += torch.dot(v1, v1)
        denom2 += torch.dot(v2, v2)
    cos = num / (torch.sqrt(denom1) * torch.sqrt(denom2))
    return cos.item()
