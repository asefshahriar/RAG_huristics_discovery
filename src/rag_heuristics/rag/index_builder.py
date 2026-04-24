from __future__ import annotations

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    SentenceTransformerEmbeddingFunction = None  # type: ignore[assignment]

from rag_heuristics.config import Settings
from rag_heuristics.core.io import read_jsonl


def _resolve_upsert_batch_size(collection: object, default: int = 1000) -> int:
    """
    Pick a safe upsert batch size across Chroma versions.

    Some builds expose `max_batch_size` from the Rust client and fail if the
    payload exceeds it. We read it when available and keep a conservative
    fallback so large corpora still index reliably.
    """
    chroma_client = getattr(collection, "_client", None)
    max_batch_size = getattr(chroma_client, "max_batch_size", None)
    if isinstance(max_batch_size, int) and max_batch_size > 0:
        return max_batch_size
    return default


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
    batch_size = _resolve_upsert_batch_size(collection)
    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=ids[start:end],
            documents=docs[start:end],
            metadatas=metadatas[start:end],
        )
    return len(ids)
