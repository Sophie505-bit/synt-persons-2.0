from typing import Dict, List, Any
from src.schema import SurveySchema


def compute_real_priors(real_answers: List[Dict[str, Any]], schema: SurveySchema) -> Dict[str, Dict[str, float]]:
    n = len(real_answers)
    if n == 0:
        return {}
    priors = {}
    for q in schema.questions:
        counts = {opt.id: 0 for opt in q.options}
        for ans in real_answers:
            selected = ans.get(q.question_id, [])
            for s in selected:
                s_id = s.split(" ")[0] if " " in s else s
                if s_id in counts:
                    counts[s_id] += 1
        priors[q.question_id] = {k: v / n for k, v in counts.items()}
    return priors


def priors_to_text(priors: Dict[str, Dict[str, float]], top_k: int = 5) -> str:
    lines = []
    for qid, dist in priors.items():
        top = sorted(dist.items(), key=lambda x: -x[1])[:top_k]
        top_str = ", ".join([f"{opt_id}={p:.0%}" for opt_id, p in top])
        lines.append(f"{qid}: {top_str}")
    return "; ".join(lines)
