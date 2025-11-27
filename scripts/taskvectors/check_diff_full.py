from pathlib import Path
import torch
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
base_path = "META-LLAMA/LLAMA-3.2-1B"
path_AB   = PROJECT_ROOT /"models/flipped_model_ba_4/AB/merged"

def is_lora_key(k):
    return any(sub in k for sub in ["q_proj", "k_proj", "v_proj"])

def main():
    base = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32, device_map={"": "cpu"})
    ab   = AutoModelForCausalLM.from_pretrained(str(path_AB), torch_dtype=torch.float32, device_map={"": "cpu"})

    sd0 = base.state_dict()
    sd_ab = ab.state_dict()

    diff_lora = 0
    diff_other = 0

    for k in sd0:
        if k not in sd_ab:
            continue
        diff = (sd_ab[k] - sd0[k]).abs().max().item()
        if diff > 1e-7:  # threshold
            if is_lora_key(k):
                diff_lora += 1
            else:
                diff_other += 1

    print(f"Changed LoRA (q/k/v) keys: {diff_lora}")
    print(f"Changed NON-LoRA keys:    {diff_other}")

if __name__ == "__main__":
    main()
