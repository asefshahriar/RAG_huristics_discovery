from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_heuristics.config import get_settings
from rag_heuristics.core.io import doc_to_dict, write_jsonl
from rag_heuristics.evaluation.benchmark import run_benchmarks, write_markdown_report
from rag_heuristics.generation.engine import train_smtt
from rag_heuristics.ingest.normalize import normalize_corpus
from rag_heuristics.rag.index_builder import build_index

app = typer.Typer(help="RAG heuristic discovery CLI")


@app.command("ingest")
def ingest(source_dir: Path = typer.Option(Path("data/raw"), exists=False)) -> None:
    settings = get_settings()
    docs = normalize_corpus(source_dir=source_dir)
    write_jsonl(settings.normalized_docs_path, [doc_to_dict(d) for d in docs])
    typer.echo(f"Ingested {len(docs)} normalized chunks into {settings.normalized_docs_path}")


@app.command("build-index")
def build_index_cmd() -> None:
    settings = get_settings()
    n = build_index(settings)
    typer.echo(f"Prepared index for {n} chunks at {settings.vector_db_path}")


@app.command("train")
def train(
    problem_type: str = typer.Option("single_machine_total_tardiness"),
    iterations: int = typer.Option(50, min=1),
) -> None:
    settings = get_settings()
    if problem_type != "single_machine_total_tardiness":
        raise typer.BadParameter("Only single_machine_total_tardiness is currently implemented.")
    result = train_smtt(settings, iterations=iterations)
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate")
def evaluate(
    problem_type: str = typer.Option("single_machine_total_tardiness"),
    iterations: int = typer.Option(40, min=10),
) -> None:
    settings = get_settings()
    if problem_type != "single_machine_total_tardiness":
        raise typer.BadParameter("Only single_machine_total_tardiness is currently implemented.")
    summary = run_benchmarks(settings, iterations=iterations)
    report_path = write_markdown_report(settings, summary)
    typer.echo(f"Wrote benchmark summary to {settings.reports_dir / 'benchmark_summary.json'}")
    typer.echo(f"Wrote markdown report to {report_path}")


if __name__ == "__main__":
    app()
