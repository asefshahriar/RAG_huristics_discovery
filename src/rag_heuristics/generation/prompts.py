from __future__ import annotations

from rag_heuristics.core.types import RetrievalChunk
from rag_heuristics.problems.smtt.spec import SPEC_DOCSTRING


def build_generation_prompt(
    problem_type: str,
    retrieved: list[RetrievalChunk],
    best_programs: list[str],
) -> str:
    snippets = "\n\n".join([f"[score={c.score:.3f}] {c.text[:600]}" for c in retrieved])
    priors = "\n\n".join(best_programs[:2]) if best_programs else "None"
    return (
        f"You are generating a Python assignment function for {problem_type}.\n"
        "Return only valid Python code defining:\n"
        "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n\n"
        f"Problem spec:\n{SPEC_DOCSTRING}\n\n"
        "Use retrieved knowledge and improve baseline quality while keeping feasibility.\n\n"
        f"Retrieved context:\n{snippets}\n\n"
        f"Best prior programs:\n{priors}\n"
    )


def fallback_candidate() -> str:
    return (
        "def assignment(processing_times: list[int], due_dates: list[int]) -> list[int]:\n"
        "    return sorted(range(len(processing_times)), key=lambda i: (due_dates[i], processing_times[i]))\n"
    )
