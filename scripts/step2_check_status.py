import time
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# The specific batch ID you provided
BATCH_ID = "batch_6915ba4d372c8190a48b39577d0697a0"

def check_status():
    print(f"--- Monitoring Batch Job: {BATCH_ID} ---")
    print("Press Ctrl+C to stop monitoring.\n")

    while True:
        try:
            # Retrieve the batch object
            batch = client.batches.retrieve(BATCH_ID)
            
            # Format the output
            status = batch.status
            completed = batch.request_counts.completed
            total = batch.request_counts.total
            failed = batch.request_counts.failed
            
            # Create a timestamp
            now = datetime.now().strftime("%H:%M:%S")

            # Clear line and print status (overwrite previous line for cleaner UI)
            # \r returns cursor to start of line
            sys.stdout.write(f"\r[{now}] Status: {status.upper()} | Progress: {completed}/{total} (Failed: {failed})")
            sys.stdout.flush()

            # Check for terminal states
            if status in ["completed", "failed", "expired", "cancelled"]:
                print(f"\n\nJob finished with status: {status}")
                if batch.output_file_id:
                    print(f"Output File ID: {batch.output_file_id}")
                if batch.error_file_id:
                    print(f"Error File ID: {batch.error_file_id}")
                
                # Auditory alert (System Beep)
                print("\a") 
                break

            # Wait 60 seconds before next check
            time.sleep(60)

        except Exception as e:
            print(f"\nError checking status: {e}")
            time.sleep(60) # Wait before retrying on error

if __name__ == "__main__":
    check_status()