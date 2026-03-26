from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pypdf import PdfReader


def load_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def iter_supported_files(source_dir: Path) -> Iterable[Path]:
    patterns = ("*.pdf", "*.txt", "*.md", "*.rst", "*.py")
    for pattern in patterns:
        yield from source_dir.rglob(pattern)
