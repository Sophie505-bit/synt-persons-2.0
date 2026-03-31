import os
import json
from typing import Optional, Dict, Any

from src.data_loader import DataLoader
from src.schema import load_schema
from src.generator import generate_population_archetypes
from src.graph_builder import build_knn_graph
from src.simulator import Simulator
from src.evaluator import Evaluator
from src.priors import compute_real_priors


def run_pipeline(
    data_file: str = "itmo_qa (3) - 8000 респондентов.xlsx",
    n: int = 1000,
    output_dir: str = "data",
    scenario_text: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(data_file)
    df = loader.get_joint_distribution()
    real_answers = loader.get_baseline_answers()
    schema = load_schema("configs/survey_schema.yaml")
    pop_file = os.path.join(output_dir, "synthetic_population.jsonl")
    baseline_file = os.path.join(output_dir, "survey_results.json")
    scenario_file = os.path.join(output_dir, "scenario_results.json")
    personas = generate_population_archetypes(df, n=n, output_file=pop_file, seed=seed)
    embeddings = [p.embedding for p in personas if p.embedding]
    neighbors = build_knn_graph(embeddings, k=10) if len(embeddings) == len(personas) else {}
    priors = compute_real_priors(real_answers, schema)
    sim = Simulator("configs/survey_schema.yaml", priors=priors)
    results = sim.run_survey(personas, neighbors_graph=neighbors, scenario=scenario_text or "")
    out_path = scenario_file if scenario_text else baseline_file
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    ev = Evaluator("configs/survey_schema.yaml")
    metrics_real = ev.compare(real_answers, results, output_dir=output_dir)
    return {"population_file": pop_file, "results_file": out_path, "metrics_real": metrics_real}
