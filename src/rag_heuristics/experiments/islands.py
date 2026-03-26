from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Island:
    island_id: int
    program_ids: list[int] = field(default_factory=list)
    best_score: float = float("inf")
    best_program_id: int | None = None


class IslandPool:
    def __init__(self, n_islands: int, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        self.islands = [Island(island_id=i) for i in range(n_islands)]

    def sample_island(self) -> Island:
        return self.rng.choice(self.islands)

    def register(self, island_id: int, program_id: int, score: float) -> None:
        island = self.islands[island_id]
        island.program_ids.append(program_id)
        if score < island.best_score:
            island.best_score = score
            island.best_program_id = program_id

    def periodic_reset(self, fraction: float = 0.5) -> None:
        n_reset = max(1, int(len(self.islands) * fraction))
        worst = sorted(self.islands, key=lambda x: x.best_score, reverse=True)[:n_reset]
        survivors = [i for i in self.islands if i not in worst and i.best_program_id is not None]
        if not survivors:
            return
        for w in worst:
            founder = self.rng.choice(survivors)
            w.program_ids = [founder.best_program_id] if founder.best_program_id is not None else []
            w.best_score = founder.best_score
            w.best_program_id = founder.best_program_id
