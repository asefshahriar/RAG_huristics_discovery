from __future__ import annotations

from pathlib import Path

from rag_heuristics.core.io import ensure_parent_dir
from rag_heuristics.generation.engine import ProgramRecord, top_program_records


def _numbered_listing(source: str) -> str:
    lines = source.strip().splitlines()
    return "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=1))


def _algorithm_section(
    algorithm_index: int,
    rec: ProgramRecord,
    *,
    problem_label: str,
) -> list[str]:
    score_txt = f"{rec.score:.6g}"
    cap = (
        f"Feasible LLM-discovered construction heuristic for **{problem_label}**. "
        f"Training metric: average total tardiness **{score_txt}** (lower is better). "
        f"Program database id `{rec.program_id}`, island `{rec.island_id}`."
    )
    block = "\n".join(
        [
            f"## Algorithm {algorithm_index} — Discovered heuristic (program #{rec.program_id})",
            "",
            f"**Caption:** {cap}",
            "",
            "**Input:** Arrays of processing times $p$ and due dates $d$ for each job "
            "(implemented as `processing_times: list[int]`, `due_dates: list[int]`).",
            "",
            "**Output:** Schedule $S$ — a permutation of job indices $\\{0,\\ldots,n-1\\}$.",
            "",
            "**Procedure:** `assignment` maps $(p, d) \\mapsto S$, analogous to "
            "Algorithms 1–2 in Çetinkaya et al. (arXiv:2510.24013).",
            "",
            "**Numbered listing** (same convention as numbered lines in the paper):",
            "",
            "```text",
            _numbered_listing(rec.source_code),
            "```",
            "",
        ]
    )
    return block.splitlines()


def build_discovered_algorithms_markdown(
    sections: list[tuple[str, list[ProgramRecord]]],
    *,
    problem_key: str,
    problem_title: str,
) -> str:
    """Render markdown with paper-style Algorithm blocks for each section."""
    header = "\n".join(
        [
            f"# Discovered heuristics — {problem_title}",
            "",
            "Presentation follows the style of Section 4.4 in "
            "Çetinkaya et al., *Discovering Heuristics with Large Language Models (LLMs) "
            "for Mixed-Integer Programs: Single-Machine Scheduling* (arXiv:2510.24013): "
            "**Caption**, **Input**, **Output**, **Procedure**, and a **numbered listing**.",
            "",
            f"Problem key: `{problem_key}`.",
            "",
            "---",
            "",
        ]
    )
    parts: list[str] = [header]
    for section_title, records in sections:
        if not records:
            parts.append(f"## {section_title}")
            parts.append("")
            parts.append("_No feasible programs in this database yet._")
            parts.append("")
            parts.append("---")
            parts.append("")
            continue
        parts.append(f"## {section_title}")
        parts.append("")
        for algo_i, rec in enumerate(records, start=1):
            parts.extend(_algorithm_section(algo_i, rec, problem_label=problem_title))
        parts.append("---")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def write_discovered_algorithms_report(
    *,
    db_path: Path,
    out_path: Path,
    problem_type: str,
    problem_title: str,
    top_k: int,
    section_title: str = "Programs database",
) -> Path:
    records = top_program_records(db_path, problem_type, k=top_k)
    body = build_discovered_algorithms_markdown(
        [(section_title, records)],
        problem_key=problem_type,
        problem_title=problem_title,
    )
    ensure_parent_dir(out_path)
    out_path.write_text(body, encoding="utf-8")
    return out_path


def write_multi_track_discovered_report(
    *,
    settings_db_path: Path,
    out_path: Path,
    problem_type: str,
    problem_title: str,
    top_k: int,
    strategies: tuple[str, ...],
) -> Path:
    """One markdown file with one subsection per seed-strategy SQLite (compare-tracks layout)."""
    sections: list[tuple[str, list[ProgramRecord]]] = []
    for strat in strategies:
        track_db = settings_db_path.with_name(f"programs_{strat}.sqlite")
        recs = top_program_records(track_db, problem_type, k=top_k)
        sections.append((f"Seed track: {strat.upper()} (`{track_db.name}`)", recs))
    body = build_discovered_algorithms_markdown(
        sections,
        problem_key=problem_type,
        problem_title=problem_title,
    )
    ensure_parent_dir(out_path)
    out_path.write_text(body, encoding="utf-8")
    return out_path
