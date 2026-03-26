from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rag_heuristics.core.types import CorpusDocument


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, items: Iterable[dict]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def doc_to_dict(doc: CorpusDocument) -> dict:
    return {
        "doc_id": doc.doc_id,
        "problem_type": doc.problem_type,
        "source_type": doc.source_type,
        "source_path": doc.source_path,
        "method_family": doc.method_family,
        "citation": doc.citation,
        "text": doc.text,
        "metadata": doc.metadata,
    }
