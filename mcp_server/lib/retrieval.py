"""FAISS-backed semantic search over unstructured text sources only.

Structured sources (stats, roster, transactions, coaching/management
changes, injuries) are served via exact lookups in loaders.py instead — this
module exists solely for free-text sources (e.g. financial narratives) where
splitting into embedding chunks is appropriate, unlike splitting an exact
structured record (the chunking-completeness failure mode flagged in the
checkpoint 3.1 design).

The index is built offline by scripts/build_faiss_index.py and loaded here
lazily, once per process.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INDEX_PATH = DATA_DIR / "index" / "faiss.index"
METADATA_PATH = DATA_DIR / "index" / "chunk_metadata.jsonl"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_state = {"index": None, "metadata": None, "model": None, "loaded": False}


def _load() -> bool:
    """Load the FAISS index + metadata + embedding model, once. Returns
    False (and leaves state empty) if no index has been built yet.
    """
    if _state["loaded"]:
        return _state["index"] is not None

    _state["loaded"] = True
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        return False

    import faiss
    from sentence_transformers import SentenceTransformer

    _state["index"] = faiss.read_index(str(INDEX_PATH))
    with open(METADATA_PATH, encoding="utf-8") as f:
        _state["metadata"] = [json.loads(line) for line in f]
    _state["model"] = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return True


def search(query: str, source: Optional[str] = None, top_k: int = 5) -> list[dict]:
    if not _load():
        return [{
            "note": "No unstructured data source has been indexed yet "
                    "(run scripts/build_faiss_index.py after adding files "
                    "under data/raw/unstructured/)."
        }]

    query_vec = _state["model"].encode([query], normalize_embeddings=True)
    # Over-fetch so post-filtering by `source` still returns up to top_k.
    fetch_k = top_k * 4 if source else top_k
    scores, indices = _state["index"].search(query_vec, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_state["metadata"]):
            continue
        chunk = _state["metadata"][idx]
        if source and chunk.get("source") != source:
            continue
        results.append({
            "text": chunk["text"],
            "source": chunk.get("source"),
            "source_file": chunk.get("source_file"),
            "score": float(score),
        })
        if len(results) >= top_k:
            break
    return results
