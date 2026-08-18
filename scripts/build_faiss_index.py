#!/usr/bin/env python3
"""Offline builder for the unstructured-text FAISS index.

Walks data/raw/unstructured/<source>/*.txt, chunks each file by paragraph,
embeds the chunks locally (no paid API), and writes:
  - data/index/faiss.index          (the vector index)
  - data/index/chunk_metadata.jsonl (text + source info per vector, in the
                                      same order as the index, so a FAISS row
                                      id maps directly to a metadata line)

Re-run this any time files under data/raw/unstructured/ change. The index is
gitignored (regenerable), so this script must be run at least once before
search_unstructured() returns real results.

Usage: python scripts/build_faiss_index.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNSTRUCTURED_DIR = REPO_ROOT / "data" / "raw" / "unstructured"
INDEX_DIR = REPO_ROOT / "data" / "index"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def chunk_text(text: str, source_file: str, source: str) -> list[dict]:
    """Paragraph-based chunking. The corpus here is small (a handful of
    narrative files), so simple paragraph splitting is sufficient — no need
    for a token-window chunker at this scale.
    """
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            chunks.append({"text": para, "source_file": source_file, "source": source})
    return chunks


def main() -> int:
    if not UNSTRUCTURED_DIR.exists():
        print(f"No unstructured data directory at {UNSTRUCTURED_DIR}; nothing to index.")
        return 0

    all_chunks = []
    for source_dir in sorted(p for p in UNSTRUCTURED_DIR.iterdir() if p.is_dir()):
        source = source_dir.name
        for txt_file in sorted(source_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            all_chunks.extend(chunk_text(text, str(txt_file.relative_to(REPO_ROOT)), source))

    if not all_chunks:
        print(f"No .txt files found under {UNSTRUCTURED_DIR}/<source>/. Nothing to index.")
        return 0

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print(f"Embedding {len(all_chunks)} chunks with {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode(
        [c["text"] for c in all_chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "chunk_metadata.jsonl", "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"Wrote {len(all_chunks)} vectors to {INDEX_DIR / 'faiss.index'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
