# step1_prepare_batch.py
import json
import os
import uuid
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PAIR_TO_VALUES = {"AB": ("A", "B"), "BC": ("B", "C"), "CA": ("C", "A")}
VALUES = {"A": "Autonomy", "B": "Honesty", "C": "Justice"}

def load_scenarios():
    with open("src/data_gen/scenarios_en.json") as f:
        return json.load(f)

def load_prompt_template():
    with open("src/data_gen/prompt_template.txt") as f:
        return f.read()

def create_request_object(pair, scenario_desc, n=20):
    """Creates a single JSON line entry for the batch API"""
    val_x, val_y = PAIR_TO_VALUES[pair]
    template = load_prompt_template()
    
    prompt = template.replace("VALUE_X", val_x)\
                     .replace("VALUE_Y", val_y)\
                     .replace("<name of VALUE_X>", VALUES[val_x])\
                     .replace("<name of VALUE_Y>", VALUES[val_y])\
                     .replace("<PAIR>", pair)\
                     .replace("N", str(n))\
                     + f"\nSCENARIO: {scenario_desc}\n"

    # Unique ID for this specific request so we can map it back later
    request_id = f"{pair}_{uuid.uuid4()}"

    # This matches the Batch API format requirements
    return {
        "custom_id": request_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.8,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"}
        }
    }

def main():
    batch_filename = "batch_tasks_input.jsonl"
    scenarios = load_scenarios()
    requests = []

    print("Generating prompt tasks...")
    
    # Generate all requests locally
    for pair in ["AB", "BC", "CA"]:
        for s in scenarios:
            # You wanted 10 batches of 20 items per scenario
            for _ in range(10): 
                req = create_request_object(pair, s["description"], n=20)
                requests.append(req)

    print(f"Created {len(requests)} tasks.")

    # 1. Write the .jsonl file
    with open(batch_filename, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    
    print(f"Saved tasks to {batch_filename}. Uploading to OpenAI...")

    # 2. Upload file
    batch_input_file = client.files.create(
        file=open(batch_filename, "rb"),
        purpose="batch"
    )
    file_id = batch_input_file.id
    print(f"File uploaded. ID: {file_id}")

    # 3. Create the Batch Job
    print("Starting batch job...")
    batch_job = client.batches.create(
        input_file_id=file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "synthetic_data_generation_v1"}
    )

    print(f"\nSUCCESS! Batch Job ID: {batch_job.id}")
    print(f"Save this ID! You will need it for step 2.")
    print("The job will take up to 24h (usually much faster).")

if __name__ == "__main__":
    main()