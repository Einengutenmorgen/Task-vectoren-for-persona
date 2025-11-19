from torch.utils.data import Dataset
from src.utils.io import load_jsonl
from src.config.values import VALUES
from src.config.tasks import TASKS

class SFTDilemmaDataset(Dataset):
    def __init__(self, path, task_name):
        """
        path: jsonl file for train/dev
        task_name: e.g. "AB", "BA"
        """
        self.data = load_jsonl(path)
        self.task_name = task_name
        self.preferred, self.other = TASKS[task_name]

    def __len__(self):
        return len(self.data)

    def build_prompt(self, item):
        preferred_value = VALUES[self.preferred]
        other_value = VALUES[self.other]

        return f"""You are given an ethical dilemma. Read the story and question, then choose between options A and B.

You must prioritize {preferred_value} over {other_value}.

Story:
{item['story']}

Question:
{item['question']}

Option A: {item['options']['A']}
Option B: {item['options']['B']}

Answer with only "A" or "B".""".strip()

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = self.build_prompt(item)
        label = item["correct_option"]  # "A" or "B"
        # We'll train LM to generate just the label after prompt
        text = prompt + "\n\n" + label
        return {"text": text}
