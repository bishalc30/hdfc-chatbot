"""Step 5 - Embedding.

Reads data/03_chunks/chunks.jsonl and writes:
  data/04_embeddings/embeddings.npy   float32 array, shape (n_chunks, 384), row i = chunk i
  data/04_embeddings/ids.json         chunk_ids in the same row order
  data/04_embeddings/manifest.json    model, dim, count, normalized, chunks file hash, timing
  data/04_embeddings/preview.json     first values of a few vectors (for the demo)

The same Embedder (get_embedder) is used at query time in step 7, so ingestion and
retrieval always use the identical model and settings.
"""
import hashlib
import json
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import get_config, path

IST = timezone(timedelta(hours=5, minutes=30))


class Embedder:
    """Thin wrapper around SentenceTransformer with normalized output."""

    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,      # unit length -> cosine similarity = dot product
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return vecs.astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    cfg = get_config()["embedding"]
    return Embedder(cfg["model_name"], cfg.get("batch_size", 64))


def file_sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run() -> np.ndarray:
    chunks_path = path("data/03_chunks/chunks.jsonl")
    if not chunks_path.exists():
        raise SystemExit("No chunks found - run `python -m src.pipeline chunk` first.")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    if not chunks:
        raise SystemExit("chunks.jsonl is empty.")

    emb = get_embedder()
    print(f"  Model: {emb.model_name} (dim={emb.dim})")
    t0 = time.time()
    vecs = emb.encode([c["text"] for c in chunks], show_progress=len(chunks) > 50)
    seconds = round(time.time() - t0, 2)

    # Sanity checks
    norms = np.linalg.norm(vecs, axis=1)
    if vecs.shape != (len(chunks), emb.dim):
        raise ValueError(f"Unexpected shape {vecs.shape}, expected ({len(chunks)}, {emb.dim})")
    if not np.all(np.isfinite(vecs)):
        raise ValueError("Embeddings contain NaN/inf values")
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(f"Embeddings are not normalized (norms {norms.min():.4f}-{norms.max():.4f})")

    out = path("data/04_embeddings")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vecs)
    with open(out / "ids.json", "w", encoding="utf-8") as f:
        json.dump([c["chunk_id"] for c in chunks], f)
    manifest = {
        "model_name": emb.model_name,
        "dim": emb.dim,
        "count": len(chunks),
        "normalized": True,
        "dtype": str(vecs.dtype),
        "chunks_sha256": file_sha256(chunks_path),   # step 6 checks this to detect stale embeddings
        "encode_seconds": seconds,
        "created_at": datetime.now(IST).isoformat(timespec="seconds"),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    preview = [{"chunk_id": c["chunk_id"], "text": c["text"][:120],
                "vector_first_8": [round(float(x), 4) for x in v[:8]]}
               for c, v in zip(chunks[:5], vecs[:5])]
    with open(out / "preview.json", "w", encoding="utf-8") as f:
        json.dump(preview, f, indent=2, ensure_ascii=False)

    print(f"  Done: {len(chunks)} vectors x {emb.dim} dims in {seconds}s")
    return vecs


if __name__ == "__main__":
    run()
