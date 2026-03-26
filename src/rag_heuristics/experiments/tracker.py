from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_heuristics.core.io import ensure_parent_dir


class ExperimentTracker:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        ensure_parent_dir(output_path)

    def log(self, payload: dict[str, Any]) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
