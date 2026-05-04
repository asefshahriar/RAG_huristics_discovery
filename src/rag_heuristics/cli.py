from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_heuristics.config import get_settings
from rag_heuristics.core.io import doc_to_dict, write_jsonl
from rag_heuristics.evaluation.benchmark import (
    run_benchmarks,
    run_seed_strategy_comparison,
    write_markdown_report,
    write_seed_strategy_report,
)
from rag_heuristics.generation.engine import train_smtt
from rag_heuristics.ingest.normalize import normalize_corpus
from rag_heuristics.rag.index_builder import build_index
from rag_heuristics.reporting.paper_algorithms import (
    write_discovered_algorithms_report,
    write_multi_track_discovered_report,
)

app = typer.Typer(help="RAG heuristic discovery CLI")

_PROBLEM_TITLES = {
    "single_machine_total_tardiness": "Single-machine total tardiness (SMTT)",
}


def _problem_title(problem_type: str) -> str:
    return _PROBLEM_TITLES.get(problem_type, problem_type)


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
    seed_strategy: str = typer.Option("mdd", help="Initial heuristic family: edd, spt, or mdd."),
    export_discovered: bool = typer.Option(
        True,
        "--export-discovered/--no-export-discovered",
        help="Write reports/discovered_algorithms.md (paper-style Algorithm blocks).",
    ),
    discovered_top_k: int = typer.Option(5, min=1, help="How many best feasible programs to export."),
    discovered_out: Path = typer.Option(
        Path("reports/discovered_algorithms.md"),
        help="Output path for the discovered-heuristics markdown report.",
    ),
) -> None:
    settings = get_settings()
    if problem_type != "single_machine_total_tardiness":
        raise typer.BadParameter("Only single_machine_total_tardiness is currently implemented.")
    if seed_strategy not in {"edd", "spt", "mdd"}:
        raise typer.BadParameter("seed_strategy must be one of: edd, spt, mdd.")
    result = train_smtt(settings, iterations=iterations, seed_strategy=seed_strategy)
    typer.echo(json.dumps(result, indent=2))
    if export_discovered:
        path = write_discovered_algorithms_report(
            db_path=settings.program_db_path,
            out_path=discovered_out,
            problem_type=problem_type,
            problem_title=_problem_title(problem_type),
            top_k=discovered_top_k,
        )
        typer.echo(f"Wrote discovered-algorithms report (paper-style) to {path}")


@app.command("export-discovered")
def export_discovered(
    problem_type: str = typer.Option("single_machine_total_tardiness"),
    top_k: int = typer.Option(5, min=1),
    out: Path = typer.Option(Path("reports/discovered_algorithms.md")),
    db_path: Path | None = typer.Option(None, help="Programs SQLite path (default: from settings)."),
    tracks: str | None = typer.Option(
        None,
        help="If set, comma-separated strategies (edd,spt,mdd) using programs_<strategy>.sqlite next to the default DB.",
    ),
) -> None:
    settings = get_settings()
    if problem_type != "single_machine_total_tardiness":
        raise typer.BadParameter("Only single_machine_total_tardiness is currently implemented.")
    active_settings = settings.model_copy(update={"program_db_path": db_path}) if db_path else settings
    if tracks:
        strat_list = tuple(s.strip().lower() for s in tracks.split(",") if s.strip())
        bad = [s for s in strat_list if s not in {"edd", "spt", "mdd"}]
        if bad:
            raise typer.BadParameter(f"Invalid strategies: {bad}; use edd, spt, mdd.")
        if not strat_list:
            raise typer.BadParameter("tracks must list at least one of: edd, spt, mdd.")
        write_multi_track_discovered_report(
            settings_db_path=active_settings.program_db_path,
            out_path=out,
            problem_type=problem_type,
            problem_title=_problem_title(problem_type),
            top_k=top_k,
            strategies=strat_list,
        )
    else:
        write_discovered_algorithms_report(
            db_path=active_settings.program_db_path,
            out_path=out,
            problem_type=problem_type,
            problem_title=_problem_title(problem_type),
            top_k=top_k,
        )
    typer.echo(f"Wrote discovered-algorithms report to {out}")


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


@app.command("compare-tracks")
def compare_tracks(
    problem_type: str = typer.Option("single_machine_total_tardiness"),
    iterations: int = typer.Option(60, min=1),
    export_discovered: bool = typer.Option(
        True,
        "--export-discovered/--no-export-discovered",
        help="Write reports/discovered_algorithms_by_seed_track.md (paper-style).",
    ),
    discovered_top_k: int = typer.Option(3, min=1),
) -> None:
    settings = get_settings()
    if problem_type != "single_machine_total_tardiness":
        raise typer.BadParameter("Only single_machine_total_tardiness is currently implemented.")
    summary = run_seed_strategy_comparison(settings, iterations=iterations)
    report_path = write_seed_strategy_report(settings, summary)
    typer.echo(f"Wrote seed strategy summary to {settings.reports_dir / 'seed_strategy_comparison.json'}")
    typer.echo(f"Wrote seed strategy report to {report_path}")
    if export_discovered:
        out_md = settings.reports_dir / "discovered_algorithms_by_seed_track.md"
        write_multi_track_discovered_report(
            settings_db_path=settings.program_db_path,
            out_path=out_md,
            problem_type=problem_type,
            problem_title=_problem_title(problem_type),
            top_k=discovered_top_k,
            strategies=("edd", "spt", "mdd"),
        )
        typer.echo(f"Wrote paper-style discovered heuristics to {out_md}")


if __name__ == "__main__":
    app()
