import json
import random
from typing import List, Dict, Any
from src.data_loader import DataLoader


class Calibrator:
    def __init__(self, data_file: str):
        self.loader = DataLoader(data_file)
        self.real_answers = self.loader.get_baseline_answers()
        self.demographics = self.loader.get_demographics()

    def get_few_shot_examples(self, n: int = 3) -> List[Dict[str, Any]]:
        indices = random.sample(range(len(self.demographics)), min(n, len(self.demographics)))
        examples = []
        for idx in indices:
            demo = self.demographics[idx]
            ans = self.real_answers[idx]
            ans_lines = []
            for qid in ["q1", "q2", "q3", "q4"]:
                selected = ans.get(qid, [])
                ids = [s.split(" ")[0] if " " in s else s for s in selected]
                if ids:
                    ans_lines.append(f"  {qid}: {ids}")
            examples.append({
                "age": demo.age,
                "gender": demo.gender,
                "region": demo.region,
                "income": demo.income,
                "education": demo.education,
                "answers": "\n".join(ans_lines),
            })
        return examples
