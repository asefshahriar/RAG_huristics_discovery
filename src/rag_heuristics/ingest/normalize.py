from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_heuristics.core.types import CorpusDocument
from rag_heuristics.ingest.loaders import iter_supported_files, load_pdf_text, load_text_file


def infer_source_type(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        if "book" in path.name.lower():
            return "book"
        return "paper"
    if path.suffix.lower() == ".py":
        return "code"
    return "text"


def infer_problem_type(text: str) -> str:
    text_l = text.lower()
    if "single machine" in text_l and "tardiness" in text_l:
        return "single_machine_total_tardiness"
    if "vehicle routing" in text_l:
        return "vehicle_routing"
    if "job-shop" in text_l or "job shop" in text_l:
        return "job_shop_scheduling"
    return "unknown"


def infer_method_family(text: str) -> str:
    t = text.lower()
    if "heuristic" in t:
        return "heuristic"
    if "dynamic programming" in t:
        return "dynamic_programming"
    if "mixed integer" in t or "mip" in t:
        return "exact_mip"
    return "general"


def normalize_corpus(source_dir: Path, chunk_size: int = 1200, chunk_overlap: int = 120) -> list[CorpusDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    docs: list[CorpusDocument] = []
    for path in iter_supported_files(source_dir):
        if path.suffix.lower() == ".pdf":
            raw_text = load_pdf_text(path)
        else:
            raw_text = load_text_file(path)
        if not raw_text:
            continue
        chunks = splitter.split_text(raw_text)
        source_type = infer_source_type(path)
        problem_type = infer_problem_type(raw_text)
        method_family = infer_method_family(raw_text)
        for i, chunk in enumerate(chunks):
            digest = hashlib.sha1(f"{path}:{i}".encode("utf-8")).hexdigest()[:12]
            docs.append(
                CorpusDocument(
                    doc_id=f"{path.stem}-{digest}",
                    problem_type=problem_type,
                    source_type=source_type,
                    source_path=str(path),
                    method_family=method_family,
                    citation=path.name,
                    text=chunk,
                    metadata={
                        "chunk_index": i,
                        "chunk_count": len(chunks),
                    },
                )
            )
    return docs
