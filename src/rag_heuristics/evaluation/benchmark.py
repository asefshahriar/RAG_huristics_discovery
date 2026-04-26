from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from rag_heuristics.config import Settings
from rag_heuristics.generation.engine import get_best_program_source, train_smtt
from rag_heuristics.generation.sandbox import compile_assignment_function
from rag_heuristics.problems.smtt.baselines import edd, mdd, mddc_like, spt
from rag_heuristics.problems.smtt.data_gen import generate_dataset
from rag_heuristics.problems.smtt.evaluator import evaluate_on_dataset


def _evaluate_source(source: str, dataset: list[tuple[list[int], list[int]]], timeout: int) -> dict:
    compiled = compile_assignment_function(source, timeout_seconds=timeout)
    if not compiled.ok or compiled.assignment_fn is None:
        return {"feasible": False, "score": float(10**12), "reason": compiled.error}
    res = evaluate_on_dataset(compiled.assignment_fn, dataset)
    return {"feasible": res.feasible, "score": res.score, "reason": res.reason}


def run_benchmarks(settings: Settings, iterations: int = 40) -> dict:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    test_instances = generate_dataset(
        n_jobs=100,
        n_instances=80,
        rdd_values=[0.2, 0.4, 0.6, 0.8],
        tf_values=[0.2, 0.4, 0.6, 0.8],
        seed=settings.random_seed + 7,
    )
    dataset = [(x.processing_times, x.due_dates) for x in test_instances]

    baselines = {
        "edd": evaluate_on_dataset(edd, dataset).score,
        "spt": evaluate_on_dataset(spt, dataset).score,
        "mdd": evaluate_on_dataset(mdd, dataset).score,
        "mddc_like": evaluate_on_dataset(mddc_like, dataset).score,
    }

    full = train_smtt(settings, iterations=iterations, use_rag=True, use_islands=True, include_code_sources=True)
    full_src = get_best_program_source(settings.program_db_path, "single_machine_total_tardiness")
    full_eval = _evaluate_source(full_src, dataset, settings.generation_timeout_seconds) if full_src else {}

    no_rag = train_smtt(settings, iterations=iterations // 2, use_rag=False, use_islands=True, include_code_sources=True)
    no_rag_src = get_best_program_source(settings.program_db_path, "single_machine_total_tardiness")
    no_rag_eval = _evaluate_source(no_rag_src, dataset, settings.generation_timeout_seconds) if no_rag_src else {}

    no_islands = train_smtt(settings, iterations=iterations // 2, use_rag=True, use_islands=False, include_code_sources=True)
    no_islands_src = get_best_program_source(settings.program_db_path, "single_machine_total_tardiness")
    no_islands_eval = _evaluate_source(no_islands_src, dataset, settings.generation_timeout_seconds) if no_islands_src else {}

    no_code = train_smtt(settings, iterations=iterations // 2, use_rag=True, use_islands=True, include_code_sources=False)
    no_code_src = get_best_program_source(settings.program_db_path, "single_machine_total_tardiness")
    no_code_eval = _evaluate_source(no_code_src, dataset, settings.generation_timeout_seconds) if no_code_src else {}

    summary = {
        "baselines": baselines,
        "full_system": {"train": full, "eval": full_eval},
        "ablation_no_rag": {"train": no_rag, "eval": no_rag_eval},
        "ablation_no_islands": {"train": no_islands, "eval": no_islands_eval},
        "ablation_no_code_retrieval": {"train": no_code, "eval": no_code_eval},
    }
    out = settings.reports_dir / "benchmark_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_seed_strategy_comparison(settings: Settings, iterations: int = 60) -> dict:
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    test_instances = generate_dataset(
        n_jobs=100,
        n_instances=80,
        rdd_values=[0.2, 0.4, 0.6, 0.8],
        tf_values=[0.2, 0.4, 0.6, 0.8],
        seed=settings.random_seed + 11,
    )
    dataset = [(x.processing_times, x.due_dates) for x in test_instances]
    baselines = {
        "edd": evaluate_on_dataset(edd, dataset).score,
        "spt": evaluate_on_dataset(spt, dataset).score,
        "mdd": evaluate_on_dataset(mdd, dataset).score,
        "mddc_like": evaluate_on_dataset(mddc_like, dataset).score,
    }
    tracks: dict[str, dict] = {}
    strategies: tuple[Literal["edd", "spt", "mdd"], ...] = ("edd", "spt", "mdd")
    for strategy in strategies:
        track_db_path = settings.program_db_path.with_name(f"programs_{strategy}.sqlite")
        track_settings = settings.model_copy(update={"program_db_path": track_db_path})
        train_result = train_smtt(
            track_settings,
            iterations=iterations,
            use_rag=True,
            use_islands=True,
            include_code_sources=True,
            seed_strategy=strategy,
        )
        best_source = get_best_program_source(track_db_path, "single_machine_total_tardiness")
        eval_result = (
            _evaluate_source(best_source, dataset, settings.generation_timeout_seconds) if best_source else {}
        )
        tracks[strategy] = {"train": train_result, "eval": eval_result}

    summary = {"baselines": baselines, "tracks": tracks, "iterations_per_track": iterations}
    out = settings.reports_dir / "seed_strategy_comparison.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def write_markdown_report(settings: Settings, summary: dict) -> Path:
    path = settings.reports_dir / "benchmark_report.md"
    baselines = summary["baselines"]
    full_score = summary["full_system"]["eval"].get("score", float("inf"))
    lines = [
        "# SMTT Benchmark Report",
        "",
        "## Baselines (avg tardiness)",
        f"- EDD: {baselines['edd']:.2f}",
        f"- SPT: {baselines['spt']:.2f}",
        f"- MDD: {baselines['mdd']:.2f}",
        f"- MDDC-like: {baselines['mddc_like']:.2f}",
        "",
        "## Discovered System",
        f"- Full pipeline score: {full_score:.2f}",
        "",
        "## Ablations",
        f"- Without RAG: {summary['ablation_no_rag']['eval'].get('score', float('inf')):.2f}",
        f"- Without islands: {summary['ablation_no_islands']['eval'].get('score', float('inf')):.2f}",
        f"- Without code retrieval: {summary['ablation_no_code_retrieval']['eval'].get('score', float('inf')):.2f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_seed_strategy_report(settings: Settings, summary: dict) -> Path:
    path = settings.reports_dir / "seed_strategy_comparison.md"
    baselines = summary["baselines"]
    tracks = summary["tracks"]
    lines = [
        "# Seed Strategy Comparison",
        "",
        "## Baselines (avg tardiness)",
        f"- EDD: {baselines['edd']:.2f}",
        f"- SPT: {baselines['spt']:.2f}",
        f"- MDD: {baselines['mdd']:.2f}",
        f"- MDDC-like: {baselines['mddc_like']:.2f}",
        "",
        "## RAG-augmented track results",
        "",
        "| Seed strategy | Score | Feasible | Reason |",
        "|---|---:|:---:|---|",
    ]
    for strategy in ("edd", "spt", "mdd"):
        eval_result = tracks.get(strategy, {}).get("eval", {})
        score = eval_result.get("score", float("inf"))
        feasible = eval_result.get("feasible", False)
        reason = str(eval_result.get("reason", "n/a")).replace("\n", " ")
        lines.append(f"| {strategy.upper()} | {score:.2f} | {str(feasible)} | {reason} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
