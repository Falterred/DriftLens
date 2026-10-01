"""Embedding/index fixture builders for the v0.1 benchmark suite (Step 34)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalized_vectors(count: int, dim: int, *, seed: int = 42) -> np.ndarray:
    """A deterministic collection of L2-normalized float32 vectors."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def faiss_flat_index(vectors: np.ndarray):
    """A reconstructable IndexFlatL2 over the given vectors."""
    import faiss

    data = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatL2(int(data.shape[1]))
    index.add(data)
    return index


def faiss_pq_index(vectors: np.ndarray, *, m: int = 4):
    """A product-quantized index trained on the given vectors."""
    import faiss

    data = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexPQ(int(data.shape[1]), m, 8)
    index.train(data)
    index.add(data)
    return index


def chroma_directory(path: Path, rows: list[dict]):
    """Create a one-collection Chroma directory from rows with id/text/embedding."""
    import chromadb
    from chromadb.config import Settings

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(path), settings=Settings(anonymized_telemetry=False)
    )
    collection = client.create_collection("docs")
    collection.add(
        ids=[row["id"] for row in rows],
        documents=[row.get("text") for row in rows],
        embeddings=[row["embedding"] for row in rows],
    )
    return client


def chroma_directory_with_pickle(path: Path) -> Path:
    """A Chroma-shaped directory carrying a stray pickle file."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "chroma.sqlite3").write_bytes(b"")
    (path / "stray.pkl").write_bytes(b"not a real pickle")
    return path
