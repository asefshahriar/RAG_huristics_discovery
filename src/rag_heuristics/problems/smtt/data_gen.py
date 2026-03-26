from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SMTTInstance:
    processing_times: list[int]
    due_dates: list[int]
    rdd: float
    tf: float


def generate_instance(n_jobs: int, rdd: float, tf: float, rng: np.random.Generator) -> SMTTInstance:
    p = rng.integers(1, 101, size=n_jobs)
    total_p = int(p.sum())
    lo = int(total_p * (1 - tf - rdd / 2))
    hi = int(total_p * (1 - tf + rdd / 2))
    if hi <= lo:
        hi = lo + 1
    d = rng.integers(lo, hi + 1, size=n_jobs)
    return SMTTInstance(processing_times=p.tolist(), due_dates=d.tolist(), rdd=rdd, tf=tf)


def generate_dataset(
    n_jobs: int,
    n_instances: int,
    rdd_values: Iterable[float],
    tf_values: Iterable[float],
    seed: int = 42,
) -> list[SMTTInstance]:
    rng = np.random.default_rng(seed)
    rdd_vals = list(rdd_values)
    tf_vals = list(tf_values)
    instances: list[SMTTInstance] = []
    for _ in range(n_instances):
        rdd = float(rng.choice(rdd_vals))
        tf = float(rng.choice(tf_vals))
        instances.append(generate_instance(n_jobs=n_jobs, rdd=rdd, tf=tf, rng=rng))
    return instances
