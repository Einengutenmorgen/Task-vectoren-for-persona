#gernerate_synthetic_data.py

import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import json
import uuid
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv

from src.utils.io import save_jsonl
from src.schema.validator import validate_dilemma

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PAIR_TO_VALUES = {
    "AB": ("A", "B"),
    "BC": ("B", "C"),
    "CA": ("C", "A")
}

VALUES = {
    "A": "Autonomy",
    "B": "Honesty",
    "C": "Justice"
}

def load_scenarios():
    with open("src/data_gen/scenarios_en.json") as f:
        return json.load(f)

def load_prompt_template():
    with open("src/data_gen/prompt_template.txt") as f:
        return f.read()

def generate_batch(pair, scenario_desc, n=20):
    val_x, val_y = PAIR_TO_VALUES[pair]
    template = load_prompt_template()
    
    prompt = template.replace("VALUE_X", val_x)\
                     .replace("VALUE_Y", val_y)\
                     .replace("<name of VALUE_X>", VALUES[val_x])\
                     .replace("<name of VALUE_Y>", VALUES[val_y])\
                     .replace("<PAIR>", pair)\
                     .replace("N", str(n))\
                     + f"\nSCENARIO: {scenario_desc}\n"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4000, # Increased slightly to prevent JSON cutoff
        response_format={"type": "json_object"} 
    )

    content = resp.choices[0].message.content

    # 1. Basic cleanup
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "")
    elif content.startswith("```"):
        content = content.replace("```", "")
    
    content = content.strip()

    # 2. Parse
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
        # Fallback: If simple load fails, return empty list to keep pipeline moving
        # rather than crashing the whole script.
        return []

    # 3. Extraction Strategy
    if isinstance(data, dict):
        # Ideally, we find the 'items' key we asked for
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        
        # Fallback: look for ANY list value
        for k, v in data.items():
            if isinstance(v, list):
                return v
                
    return [] # Return empty if structure is totally wrong

def main():
    print("Generating synthetic data...")
    save_dir = Path("data/raw/synthetic_en")
    save_dir.mkdir(parents=True, exist_ok=True)

    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios.")
    for pair in ["AB", "BC", "CA"]:
        all_items = []

        for s in scenarios:
            for _ in range(10):  # 20 scenarios × 10 batches × 20 items = 4000 total
                items = generate_batch(pair, s["description"], n=20)
                for item in items:
                    # assign UUID
                    item["id"] = str(uuid.uuid4())

                    ok = validate_dilemma(item)
                    if ok is True:
                        all_items.append(item)
            print(f"Pair {pair}, Scenario {s['id']}: Collected {len(all_items)} valid items so far.")

        save_jsonl(all_items, save_dir / f"{pair}.jsonl")
        print(pair, "total samples:", len(all_items))

if __name__ == "__main__":
    main()
