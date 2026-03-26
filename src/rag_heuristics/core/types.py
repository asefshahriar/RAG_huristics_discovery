from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CorpusDocument:
    doc_id: str
    problem_type: str
    source_type: str
    source_path: str
    method_family: str
    citation: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalChunk:
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any]


@dataclass
class CandidateProgram:
    source_code: str
    score: float
    feasible: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)
