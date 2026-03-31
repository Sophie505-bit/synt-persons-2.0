import json
import typer
from typing import Optional

app = typer.Typer()


@app.command("run-pipeline")
def run_pipeline_cmd(
    data_file: str = typer.Option("itmo_qa (3) - 8000 респондентов.xlsx", "--data-file"),
    n: int = typer.Option(200, "--n"),
    scenario_file: Optional[str] = typer.Option(None, "--scenario-file"),
    output_dir: str = typer.Option("data", "--output-dir"),
    seed: int = typer.Option(42, "--seed"),
):
    from src.pipeline import run_pipeline
    scenario_text = None
    if scenario_file:
        with open(scenario_file, "r", encoding="utf-8") as f:
            scenario_text = f.read()
    result = run_pipeline(data_file=data_file, n=n, output_dir=output_dir, scenario_text=scenario_text, seed=seed)
    print("pipeline done")
    print(json.dumps(result["metrics_real"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
