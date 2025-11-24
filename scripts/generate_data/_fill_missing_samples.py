import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import uuid
import random
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

# Reuse your existing utility functions
from src.utils.io import save_jsonl, load_jsonl
from src.schema.validator import validate_dilemma

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- CONFIGURATION ---
TARGET_COUNT = 4000
BATCH_SIZE = 5 
DATA_DIR = Path("data/raw/synthetic_en")

# Mapping pairs to their letter codes
PAIR_TO_VALUES = {
    "AB": ("A", "B"),
    "BC": ("B", "C"),
    "CA": ("C", "A")
}

# Mapping letter codes to actual value names
VALUES = {
    "A": "Autonomy",
    "B": "Honesty",
    "C": "Justice"
}

def load_scenarios():
    # Adjust path if necessary depending on where you run this script from
    try:
        with open("src/data_gen/scenarios_en.json") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: Could not find src/data_gen/scenarios_en.json")
        sys.exit(1)

def load_prompt_template():
    try:
        with open("src/data_gen/prompt_template.txt") as f:
            return f.read()
    except FileNotFoundError:
        print("Error: Could not find src/data_gen/prompt_template.txt")
        sys.exit(1)

def generate_batch(pair, scenario_desc, n=5):
    val_x, val_y = PAIR_TO_VALUES[pair]
    template = load_prompt_template()
    
    # Replace placeholders with specific values for this pair
    prompt = template.replace("VALUE_X", val_x)\
                     .replace("VALUE_Y", val_y)\
                     .replace("<name of VALUE_X>", VALUES[val_x])\
                     .replace("<name of VALUE_Y>", VALUES[val_y])\
                     .replace("<PAIR>", pair)\
                     .replace("N", str(n))\
                     + f"\nSCENARIO: {scenario_desc}\n"

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9, 
            max_tokens=4000,
            response_format={"type": "json_object"} 
        )
        content = resp.choices[0].message.content
    except Exception as e:
        print(f"  [!] API Error: {e}")
        return []

    # Clean up markdown if present
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    elif content.startswith("```"):
        content = content.replace("```", "")
    content = content.strip()

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        print("  [!] JSON Decode Error")
        return []

    # Extract list items regardless of wrapper key
    if isinstance(data, dict):
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        for k, v in data.items():
            if isinstance(v, list):
                return v
    return []

def main():
    print(f"=== Starting Top-Up Process (Target: {TARGET_COUNT}) ===")
    print(f"Scanning folder: {DATA_DIR.resolve()}")
    
    scenarios = load_scenarios()

    for pair in ["AB", "BC", "CA"]:
        file_path = DATA_DIR / f"{pair}.jsonl"
        
        # If file doesn't exist, create empty list
        if not file_path.exists():
            print(f"[{pair}] File NOT found at {file_path}. Starting fresh.")
            existing_items = []
        else:
            existing_items = load_jsonl(file_path)
        
        current_count = len(existing_items)
        print(f"[{pair}] Current count: {current_count}")

        if current_count >= TARGET_COUNT:
            print(f"[{pair}] Target met. Skipping.")
            continue

        needed = TARGET_COUNT - current_count
        print(f"[{pair}] Missing {needed} samples. Generating...")

        # Determine expected values for validation
        # e.g. for CA: val_x='C' (Justice), val_y='A' (Autonomy)
        val_x_key, val_y_key = PAIR_TO_VALUES[pair]
        expected_value_names = {VALUES[val_x_key], VALUES[val_y_key]}

        while len(existing_items) < TARGET_COUNT:
            s = random.choice(scenarios)
            
            new_batch = generate_batch(pair, s["description"], n=BATCH_SIZE)
            
            valid_new_items = 0
            for item in new_batch:
                if len(existing_items) >= TARGET_COUNT:
                    break

                # --- SAFETY CHECK START ---
                # Verify that the generated item actually contains the correct values.
                # This prevents "AB" (Honesty/Autonomy) data from leaking into "CA" (Justice/Autonomy).
                
                mapping = item.get("value_mapping", {})
                generated_values = set(mapping.values())
                
                # Check intersection: Do we have BOTH expected values?
                if not expected_value_names.issubset(generated_values):
                    # print(f"    [!] SKIPPING invalid generation. Expected {expected_value_names}, got {generated_values}")
                    continue
                # --- SAFETY CHECK END ---

                item["id"] = str(uuid.uuid4())
                
                # Run standard schema validation
                if validate_dilemma(item):
                    existing_items.append(item)
                    valid_new_items += 1
            
            print(f"    -> Added {valid_new_items} items. Total: {len(existing_items)}/{TARGET_COUNT}")

        # Overwrite/Save the file
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(existing_items, file_path)
        print(f"[{pair}] UPDATED. Saved {len(existing_items)} samples to {file_path}\n")

if __name__ == "__main__":
    main()