# step2_retrieve_results.py
import json
import os
import uuid
import sys
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from src.utils.io import save_jsonl
from src.schema.validator import validate_dilemma

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# PASTE YOUR BATCH ID HERE FROM STEP 1
BATCH_JOB_ID = "batch_6915ba4d372c8190a48b39577d0697a0" #save from terminal output of step 1

def parse_json_content(content_str):
    """Robust JSON parser for the model output"""
    content_str = content_str.strip()
    if content_str.startswith("```json"):
        content_str = content_str.replace("```json", "").replace("```", "")
    elif content_str.startswith("```"):
        content_str = content_str.replace("```", "")
    
    try:
        data = json.loads(content_str)
        # Logic to extract list items regardless of key name
        if isinstance(data, dict):
            if "items" in data and isinstance(data["items"], list):
                return data["items"]
            for k, v in data.items():
                if isinstance(v, list):
                    return v
        return []
    except json.JSONDecodeError:
        return []

def main():
    # 1. Check Status
    batch = client.batches.retrieve(BATCH_JOB_ID)
    print(f"Batch Status: {batch.status}")

    if batch.status != "completed":
        print("Batch is not ready yet. Try again later.")
        return

    if not batch.output_file_id:
        print("Batch completed but no output file ID found (maybe all failed?).")
        return

    # 2. Download Results
    print("Downloading results...")
    file_response = client.files.content(batch.output_file_id)
    raw_responses = file_response.text.strip().split('\n')

    print(f"Downloaded {len(raw_responses)} responses. Processing...")

    save_dir = Path("data/raw/synthetic_en")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Group items by their pair type (AB, BC, CA) based on the custom_id we set in step 1
    # custom_id format was: "AB_uuid"
    grouped_items = {"AB": [], "BC": [], "CA": []}

    for line in raw_responses:
        res = json.loads(line)
        custom_id = res['custom_id']
        pair_type = custom_id.split('_')[0] # Extracts AB, BC, or CA

        # Check if the specific request succeeded
        if res['response']['status_code'] != 200:
            print(f"Request {custom_id} failed.")
            continue

        # Extract the model's actual text generation
        assistant_msg = res['response']['body']['choices'][0]['message']['content']
        
        # Parse the synthetic data
        items = parse_json_content(assistant_msg)

        # Validate and collect
        for item in items:
            item["id"] = str(uuid.uuid4())
            # if validate_dilemma(item):  # Uncomment if you have this
            grouped_items[pair_type].append(item)

    # 3. Save to files
    for pair, items in grouped_items.items():
        if items:
            # output_path = save_dir / f"{pair}.jsonl"
            # save_jsonl(items, output_path) # Uncomment if you have this
            print(f"Saved {len(items)} items for {pair}")
            
            # Simple fallback save if you don't have save_jsonl imported right now
            with open(save_dir / f"{pair}.jsonl", "w") as f:
                for item in items:
                    f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()