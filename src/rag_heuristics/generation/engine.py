from __future__ import annotations

import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Literal

from rag_heuristics.config import Settings
from rag_heuristics.experiments.islands import IslandPool
from rag_heuristics.experiments.tracker import ExperimentTracker
from rag_heuristics.generation.prompts import build_generation_prompt, fallback_candidate
from rag_heuristics.generation.sandbox import compile_assignment_function
from rag_heuristics.problems.smtt.baselines import edd, mdd, mddc_like, spt
from rag_heuristics.problems.smtt.data_gen import generate_dataset
from rag_heuristics.problems.smtt.evaluator import EvaluationResult, evaluate_on_dataset
from rag_heuristics.rag.retriever import ProblemAwareRetriever

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


def init_program_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS programs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                problem_type TEXT NOT NULL,
                source_code TEXT NOT NULL,
                score REAL NOT NULL,
                feasible INTEGER NOT NULL,
                reason TEXT NOT NULL,
                island_id INTEGER NOT NULL,
                prompt TEXT
            )
            """
        )
        con.commit()


def insert_program(
    db_path: Path,
    problem_type: str,
    source_code: str,
    result: EvaluationResult,
    island_id: int,
    prompt: str,
) -> int:
    with sqlite3.connect(db_path) as con:
        cur = con.execute(
            """
            INSERT INTO programs (created_at, problem_type, source_code, score, feasible, reason, island_id, prompt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (time.time(), problem_type, source_code, result.score, int(result.feasible), result.reason, island_id, prompt),
        )
        con.commit()
        return int(cur.lastrowid)


def top_programs(db_path: Path, problem_type: str, k: int = 5) -> list[str]:
    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            """
            SELECT source_code FROM programs
            WHERE problem_type = ? AND feasible = 1
            ORDER BY score ASC, created_at DESC
            LIMIT ?
            """,
            (problem_type, k),
        ).fetchall()
    return [r[0] for r in rows]


def _openai_generate(prompt: str, model_name: str) -> str:
    if OpenAI is None:
        raise RuntimeError("openai package missing")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    completion = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0.7,
    )
    return completion.output_text.strip()


def _fallback_generate(rng: random.Random) -> str:
    library = [
        fallback_candidate(),
        (
            "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
            "    return sorted(range(len(processing_times)), key=lambda i: due_dates[i])\n"
        ),
        (
            "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
            "    unscheduled = set(range(len(processing_times)))\n"
            "    schedule = []\n"
            "    t = 0\n"
            "    while unscheduled:\n"
            "        j = min(unscheduled, key=lambda x: max(due_dates[x], t + processing_times[x]))\n"
            "        schedule.append(j)\n"
            "        unscheduled.remove(j)\n"
            "        t += processing_times[j]\n"
            "    return schedule\n"
        ),
    ]
    return rng.choice(library)


def _seed_strategy_source(seed_strategy: Literal["edd", "spt", "mdd"]) -> str:
    if seed_strategy == "edd":
        return (
            "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
            "    return sorted(range(len(processing_times)), key=lambda i: (due_dates[i], processing_times[i]))\n"
        )
    if seed_strategy == "spt":
        return (
            "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
            "    return sorted(range(len(processing_times)), key=lambda i: processing_times[i])\n"
        )
    return (
        "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
        "    unscheduled = set(range(len(processing_times)))\n"
        "    schedule = []\n"
        "    t = 0\n"
        "    while unscheduled:\n"
        "        j = min(unscheduled, key=lambda x: max(due_dates[x], t + processing_times[x]))\n"
        "        schedule.append(j)\n"
        "        unscheduled.remove(j)\n"
        "        t += processing_times[j]\n"
        "    return schedule\n"
    )


def _seed_query(seed_strategy: Literal["edd", "spt", "mdd"]) -> str:
    return f"single machine total tardiness heuristics {seed_strategy.upper()} assignment"


def train_smtt(
    settings: Settings,
    iterations: int = 50,
    n_islands: int = 10,
    use_rag: bool = True,
    use_islands: bool = True,
    include_code_sources: bool = True,
    seed_strategy: Literal["edd", "spt", "mdd"] = "mdd",
) -> dict:
    init_program_db(settings.program_db_path)
    tracker = ExperimentTracker(settings.reports_dir / "training_log.jsonl")
    retriever = ProblemAwareRetriever(settings)
    islands = IslandPool(n_islands=n_islands, seed=settings.random_seed)
    rng = random.Random(settings.random_seed)

    train_instances = generate_dataset(
        n_jobs=25,
        n_instances=200,
        rdd_values=[0.2, 0.4, 0.6, 0.8, 1.0],
        tf_values=[0.2, 0.4, 0.6, 0.8, 1.0],
        seed=settings.random_seed,
    )
    dataset = [(x.processing_times, x.due_dates) for x in train_instances]

    baseline_results = {
        "edd": evaluate_on_dataset(edd, dataset).score,
        "spt": evaluate_on_dataset(spt, dataset).score,
        "mdd": evaluate_on_dataset(mdd, dataset).score,
        "mddc_like": evaluate_on_dataset(mddc_like, dataset).score,
    }
    seed_program = _seed_strategy_source(seed_strategy)

    for i in range(iterations):
        island = islands.sample_island() if use_islands else islands.islands[0]
        chunks = []
        if use_rag:
            source_types = ["paper", "book", "text"]
            if include_code_sources:
                source_types.append("code")
            chunks = retriever.retrieve(
                query=_seed_query(seed_strategy),
                problem_type="single_machine_total_tardiness",
                top_k=settings.default_top_k,
                source_types=source_types,
            )
        priors = top_programs(settings.program_db_path, "single_machine_total_tardiness", k=3)
        if not priors:
            priors = [seed_program]
        else:
            priors = [seed_program, *priors]
        prompt = build_generation_prompt("single_machine_total_tardiness", chunks, priors)

        try:
            source = _openai_generate(prompt, settings.model_name)
        except Exception:
            source = rng.choice([seed_program, _fallback_generate(rng)])

        compiled = compile_assignment_function(source, timeout_seconds=settings.generation_timeout_seconds)
        if not compiled.ok or compiled.assignment_fn is None:
            result = EvaluationResult(
                feasible=False,
                tardiness=10**12,
                score=10**12,
                reason=f"compile_error:{compiled.error}",
            )
        else:
            result = evaluate_on_dataset(compiled.assignment_fn, dataset)

        program_id = insert_program(
            db_path=settings.program_db_path,
            problem_type="single_machine_total_tardiness",
            source_code=source,
            result=result,
            island_id=island.island_id,
            prompt=prompt,
        )
        islands.register(island_id=island.island_id, program_id=program_id, score=result.score)

        if use_islands and (i + 1) % 20 == 0:
            islands.periodic_reset(fraction=0.5)

        tracker.log(
            {
                "iteration": i + 1,
                "island_id": island.island_id,
                "program_id": program_id,
                "score": result.score,
                "feasible": result.feasible,
                "reason": result.reason,
                "baseline_scores": baseline_results,
                "retrieved_chunks": [c.doc_id for c in chunks],
                "seed_strategy": seed_strategy,
            }
        )

    best = top_programs(settings.program_db_path, "single_machine_total_tardiness", k=1)
    return {
        "iterations": iterations,
        "baseline_scores": baseline_results,
        "best_program_found": bool(best),
        "use_rag": use_rag,
        "use_islands": use_islands,
        "include_code_sources": include_code_sources,
        "seed_strategy": seed_strategy,
    }


def get_best_program_source(db_path: Path, problem_type: str) -> str | None:
    best = top_programs(db_path, problem_type, k=1)
    return best[0] if best else None
