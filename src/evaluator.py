import os
from typing import List, Dict, Any
from src.schema import load_schema


class Evaluator:
    def __init__(self, schema_path: str):
        self.schema = load_schema(schema_path)

    def aggregate_answers(self, answers: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        n = len(answers)
        if n == 0:
            return {}
        agg = {}
        for q in self.schema.questions:
            counts = {opt.id: 0 for opt in q.options}
            for ans in answers:
                selected = ans.get(q.question_id, [])
                for s in selected:
                    s_id = s.split(" ")[0] if " " in s else s
                    if s_id in counts:
                        counts[s_id] += 1
            agg[q.question_id] = {k: v / n for k, v in counts.items()}
        return agg

    def compute_tvd(self, dist_a, dist_b, option_ids):
        return 0.5 * sum(abs(dist_a.get(oid, 0) - dist_b.get(oid, 0)) for oid in option_ids)

    def compare(self, real_answers, synth_answers, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        real_agg = self.aggregate_answers(real_answers)
        synth_agg = self.aggregate_answers(synth_answers)
        metrics = {}
        for q in self.schema.questions:
            qid = q.question_id
            option_ids = [opt.id for opt in q.options]
            tvd = self.compute_tvd(real_agg.get(qid, {}), synth_agg.get(qid, {}), option_ids)
            metrics[qid] = {"TVD": tvd}
        return metrics

    def compare_synthetic(self, base_answers, scenario_answers, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        base_agg = self.aggregate_answers(base_answers)
        scen_agg = self.aggregate_answers(scenario_answers)
        metrics = {}
        for q in self.schema.questions:
            qid = q.question_id
            option_ids = [opt.id for opt in q.options]
            tvd = self.compute_tvd(base_agg.get(qid, {}), scen_agg.get(qid, {}), option_ids)
            metrics[qid] = {"TVD": tvd}
        return metrics
