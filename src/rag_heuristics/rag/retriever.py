from __future__ import annotations

from typing import Any

try:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
except ImportError:  # pragma: no cover
    chromadb = None  # type: ignore[assignment]
    SentenceTransformerEmbeddingFunction = None  # type: ignore[assignment]

from rag_heuristics.config import Settings
from rag_heuristics.core.io import read_jsonl
from rag_heuristics.core.types import RetrievalChunk


class ProblemAwareRetriever:
    def __init__(self, settings: Settings, collection_name: str = "heuristics_corpus") -> None:
        self._settings = settings
        self._fallback_rows = read_jsonl(settings.normalized_docs_path)
        self._collection = None
        if chromadb is not None and SentenceTransformerEmbeddingFunction is not None:
            try:
                client = chromadb.PersistentClient(path=str(settings.vector_db_path))
                self._collection = client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=SentenceTransformerEmbeddingFunction(model_name=settings.embedding_model),
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                self._collection = None

    def retrieve(
        self,
        query: str,
        problem_type: str,
        top_k: int | None = None,
        source_types: list[str] | None = None,
    ) -> list[RetrievalChunk]:
        if self._collection is None:
            return self._fallback_retrieve(query, problem_type, top_k=top_k, source_types=source_types)
        where: dict[str, Any] = {"problem_type": problem_type}
        if source_types:
            where = {"$and": [where, {"source_type": {"$in": source_types}}]}
        k = top_k or self._settings.default_top_k
        result = self._collection.query(query_texts=[query], where=where, n_results=k)
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        chunks: list[RetrievalChunk] = []
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists, strict=False):
            # Hybrid-style bias: prioritize direct heuristic keyword matches.
            bonus = 0.03 if "heuristic" in text.lower() else 0.0
            score = float(1.0 - dist + bonus)
            chunks.append(RetrievalChunk(doc_id=doc_id, text=text, score=score, metadata=meta))
        chunks.sort(key=lambda c: c.score, reverse=True)
        return chunks

    def _fallback_retrieve(
        self,
        query: str,
        problem_type: str,
        top_k: int | None = None,
        source_types: list[str] | None = None,
    ) -> list[RetrievalChunk]:
        tokens = [t for t in query.lower().split() if len(t) > 2]
        rows = [r for r in self._fallback_rows if r.get("problem_type") in {problem_type, "unknown"}]
        if source_types:
            rows = [r for r in rows if r.get("source_type") in set(source_types)]
        scored: list[RetrievalChunk] = []
        for row in rows:
            text = row.get("text", "")
            text_l = text.lower()
            overlap = sum(1 for tok in tokens if tok in text_l)
            if overlap == 0:
                continue
            score = float(overlap / max(1, len(tokens)))
            scored.append(
                RetrievalChunk(
                    doc_id=row["doc_id"],
                    text=text,
                    score=score,
                    metadata=row.get("metadata", {}),
                )
            )
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[: (top_k or self._settings.default_top_k)]
