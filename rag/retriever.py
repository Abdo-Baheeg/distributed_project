"""
FAISS-based context retrieval with sentence-transformers embeddings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def _chunk_text(text: str, max_chars: int = 400) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start = end
    return chunks


class FaissRetriever:
    def __init__(
        self,
        index_dir: str | Path | None = None,
        embed_model_name: str | None = None,
    ) -> None:
        _root = Path(__file__).resolve().parent.parent
        default_dir = _root / "rag_index"
        self.index_dir = Path(index_dir or os.environ.get("RAG_INDEX_DIR", str(default_dir)))
        self.embed_model_name = embed_model_name or _DEFAULT_MODEL
        self._model = None
        self._index: faiss.Index | None = None
        self._chunks: list[str] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embed_model_name)
        return self._model

    def ensure_loaded(self) -> None:
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "meta.json"
        corpus_path = Path(__file__).parent / "sample_corpus.txt"

        self.index_dir.mkdir(parents=True, exist_ok=True)

        if index_path.exists() and meta_path.exists():
            self._load_index(index_path, meta_path)
            return

        logger.info("Building FAISS index from %s", corpus_path)
        text = corpus_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        if not chunks:
            chunks = ["empty corpus"]
        model = self._get_model()
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, str(index_path))
        meta_path.write_text(json.dumps({"chunks": chunks}, ensure_ascii=False), encoding="utf-8")
        self._index = index
        self._chunks = chunks

    def _load_index(self, index_path: Path, meta_path: Path) -> None:
        self._index = faiss.read_index(str(index_path))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = meta["chunks"]

    def retrieve_context(self, query: str, k: int = 4) -> list[str]:
        """Return top-k relevant text snippets for the query."""
        self.ensure_loaded()
        if self._index is None or not self._chunks:
            return []

        model = self._get_model()
        q = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q)
        k = min(k, len(self._chunks))
        scores, indices = self._index.search(q.astype(np.float32), k)
        out: list[str] = []
        for idx in indices[0]:
            if 0 <= int(idx) < len(self._chunks):
                out.append(self._chunks[int(idx)])
        return out


def retrieve_context(query: str, k: int = 4) -> list[str]:
    """Module-level helper using env RAG_INDEX_DIR."""
    r = FaissRetriever()
    return r.retrieve_context(query, k=k)


__all__ = ["FaissRetriever", "retrieve_context"]
