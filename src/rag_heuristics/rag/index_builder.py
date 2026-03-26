from __future__ import annotations

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    SentenceTransformerEmbeddingFunction = None  # type: ignore[assignment]

from rag_heuristics.config import Settings
from rag_heuristics.core.io import read_jsonl


def build_index(settings: Settings, collection_name: str = "heuristics_corpus") -> int:
    rows = read_jsonl(settings.normalized_docs_path)
    if chromadb is None or SentenceTransformerEmbeddingFunction is None:
        # Fallback mode for environments where vector dependencies are not installed.
        return len(rows)
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(settings.vector_db_path))
    try:
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model)
    except Exception:
        # If sentence-transformers backend is unavailable, keep fallback mode active.
        return len(rows)
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    if not rows:
        return 0
    ids = [r["doc_id"] for r in rows]
    docs = [r["text"] for r in rows]
    metadatas = []
    for r in rows:
        m = dict(r.get("metadata", {}))
        m["problem_type"] = r["problem_type"]
        m["source_type"] = r["source_type"]
        m["method_family"] = r["method_family"]
        m["citation"] = r["citation"]
        m["source_path"] = r["source_path"]
        metadatas.append(m)
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return len(ids)
