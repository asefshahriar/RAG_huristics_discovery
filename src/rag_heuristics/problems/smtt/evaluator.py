from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable


PENALTY_SCORE = 10**12


@dataclass
class EvaluationResult:
    feasible: bool
    tardiness: float
    score: float
    reason: str


def total_tardiness(processing_times: list[int], due_dates: list[int], schedule: list[int]) -> int:
    t = 0
    total = 0
    for job in schedule:
        t += processing_times[job]
        total += max(0, t - due_dates[job])
    return total


def validate_schedule(n_jobs: int, schedule: list[int]) -> tuple[bool, str]:
    if len(schedule) != n_jobs:
        return False, "length_mismatch"
    if any(not isinstance(x, int) for x in schedule):
        return False, "non_integer_indices"
    if any(x < 0 or x >= n_jobs for x in schedule):
        return False, "index_out_of_bounds"
    if len(set(schedule)) != n_jobs:
        return False, "duplicate_or_missing_jobs"
    return True, "ok"


def evaluate_assignment(
    assignment_fn: Callable[[list[int], list[int]], list[int]],
    processing_times: list[int],
    due_dates: list[int],
) -> EvaluationResult:
    p_before = copy.deepcopy(processing_times)
    d_before = copy.deepcopy(due_dates)
    try:
        schedule = assignment_fn(processing_times, due_dates)
    except Exception as exc:  # noqa: BLE001
        return EvaluationResult(feasible=False, tardiness=PENALTY_SCORE, score=PENALTY_SCORE, reason=f"runtime_error:{exc}")

    if processing_times != p_before or due_dates != d_before:
        return EvaluationResult(feasible=False, tardiness=PENALTY_SCORE, score=PENALTY_SCORE, reason="inputs_mutated")

    feasible, reason = validate_schedule(len(processing_times), schedule)
    if not feasible:
        return EvaluationResult(feasible=False, tardiness=PENALTY_SCORE, score=PENALTY_SCORE, reason=reason)

    tardiness = float(total_tardiness(processing_times, due_dates, schedule))
    return EvaluationResult(feasible=True, tardiness=tardiness, score=tardiness, reason="ok")


def evaluate_on_dataset(
    assignment_fn: Callable[[list[int], list[int]], list[int]],
    dataset: list[tuple[list[int], list[int]]],
) -> EvaluationResult:
    scores: list[float] = []
    for p, d in dataset:
        res = evaluate_assignment(assignment_fn, p.copy(), d.copy())
        if not res.feasible:
            return res
        scores.append(res.score)
    avg = sum(scores) / max(1, len(scores))
    return EvaluationResult(feasible=True, tardiness=avg, score=avg, reason="ok")
