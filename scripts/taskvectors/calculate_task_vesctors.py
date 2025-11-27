from datetime import datetime
import sys
import os
from pathlib import Path

# 1. Setup Path & Cuda (Keep this as is)
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent
sys.path.append(str(project_root))

# Set device before importing torch usually avoids some initialization warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch

from src.utils.eval_and_gridsearch import run_task_vector_alignment

if __name__ == "__main__":

    print(f'Project Root detected as: {project_root}')
    print('Checking paths...')


    base_path = "META-LLAMA/LLAMA-3.2-1B"
    
    
    path_AB = project_root / "models/sft_seed/AB/merged"
    if not path_AB.exists():
        print(f'ERROR: Did not find file: {path_AB}')
        raise FileNotFoundError(f"Missing: {path_AB}")

    path_BC = project_root / "models/sft_seed/BC/merged"
    if not path_BC.exists():
        print(f'ERROR: Did not find file: {path_BC}')
        raise FileNotFoundError(f"Missing: {path_BC}")

    path_CB = project_root / "models/sft_seed/CB/merged"
    if not path_CB.exists():        
        print(f'ERROR: Did not find file: {path_CB}')
        raise FileNotFoundError(f"Missing: {path_CB}")

    dev_jsonl = project_root / "data/sft_taskvectors/BA/dev.jsonl"
    if not dev_jsonl.exists():
        print(f'ERROR: Did not find file: {dev_jsonl}')
        raise FileNotFoundError(f"Missing: {dev_jsonl}")


    
    save_tmp_dir = project_root / "tmp_taskvec_search"
    save_tmp_dir.mkdir(parents=True, exist_ok=True)

    save_final_dir = project_root / "flipped_AB_to_BA_model_with_seed"
    save_final_dir.mkdir(parents=True, exist_ok=True)

    # --- Device Setup ---
    device = "cuda"
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device: {device}")
    
    date = datetime.today()
    print(f'Starting: {date}')

    # --- Run Function ---
    run_task_vector_alignment(
        base_path=base_path,
        path_AB=str(path_AB),    # Convert Path objects to strings for the function
        path_BC=str(path_BC),
        path_CB=str(path_CB),    # FIX 4: Corrected typo (was path_BC)
        dev_jsonl=str(dev_jsonl),
        save_tmp_dir=str(save_tmp_dir),
        save_final_dir=str(save_final_dir),
        device=device
    )

    print('DONE SUCCESSFULLY')