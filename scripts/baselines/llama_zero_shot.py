import os
import sys
import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.io import load_jsonl

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Define Inputs and Outputs
INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "en" / "tasks"
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "en" / "llama"

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
SAFE_MODEL_NAME = MODEL_NAME.replace("/", "_")

DEVICE = "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else \
         "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
# Set padding side to left for decoder-only inference
tokenizer.padding_side = "left" 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN).to(DEVICE)

# Define terminators for Llama 3 (EOS token and EOT token)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

def format_prompt_messages(item):
    """
    Returns a list of messages for the chat template
    """
    user_content = f"""
You will be shown a dilemma. Choose A or B.

Story:
{item['story']}

Question:
{item['question']}

Options:
A) {item['options']['A']}
B) {item['options']['B']}

Respond only with the letter A or B.
""".strip()
    
    return [
        {"role": "user", "content": user_content}
    ]

def extract_answer(text):
    text = text.strip().upper()
    # Handle cases like "Answer: A" or "A)"
    if len(text) > 0:
        if text.startswith("A"): return "A"
        if text.startswith("B"): return "B"
    # Fallback scan
    for ch in text:
        if ch in ("A", "B"):
            return ch
    return None

def evaluate_task(task):
    input_file = INPUT_DIR / task / "test.jsonl"
    
    output_dir = RESULTS_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{SAFE_MODEL_NAME}_test.jsonl"

    items = load_jsonl(input_file)
    results = []
    correct = 0
    valid_total = 0
    count_A = 0
    count_B = 0

    print(f"\nEvaluating {task} -> saving to {output_file.name}")

    for idx, item in tqdm(enumerate(items), total=len(items)):
        
        # 1. Create Message Structure
        messages = format_prompt_messages(item)
        
        # 2. Apply Chat Template
        # add_generation_prompt=True is CRITICAL. It adds the "assistant" header.
        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(DEVICE)

        # 3. Generate
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=5, # Give it a little breathing room
                temperature=0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators,
                do_sample=False
            )

        # 4. Decode ONLY new tokens
        generated_ids = output[0][input_ids.shape[1]:]
        raw_response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        answer = extract_answer(raw_response)

        is_correct = False
        if answer:
            valid_total += 1
            if answer == "A": count_A += 1
            elif answer == "B": count_B += 1
            
            if answer == item["correct_option"]:
                correct += 1
                is_correct = True

        result_entry = {
            "index": idx,
            "source_id": item.get("id"),
            "model": MODEL_NAME,
            "task": task,
            "prediction": answer,
            "correct_option": item.get("correct_option"),
            "is_correct": is_correct,
            "raw_content": raw_response
        }
        results.append(result_entry)

    # Save
    with open(output_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    # Stats
    if valid_total > 0:
        acc = correct / valid_total * 100
        bias_a = count_A / valid_total * 100
        bias_b = count_B / valid_total * 100
        print(f"{task}: Accuracy = {acc:.2f}% ({correct}/{valid_total})")
        print(f"  Bias: A={bias_a:.1f}%, B={bias_b:.1f}%")
    else:
        print(f"{task}: No valid answers extracted.")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for task in ["AB", "BA", "BC", "CB", "CA", "AC"]:
        evaluate_task(task)

if __name__ == "__main__":
    main()