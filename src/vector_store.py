"""Step 6 - Store data in the vector DB (ChromaDB).

Reads data/03_chunks/chunks.jsonl + data/04_embeddings/* and writes:
  data/05_vectordb/                   persistent ChromaDB collection
  data/05_vectordb/index_report.json  counts, checks and a smoke-test query (for the demo)

Vectors come from step 5 (we pass them in), so Chroma's own embedding function is
disabled. The collection uses cosine distance and records the embedding model name;
step 7 refuses to query if that name differs from config.yaml.
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone

import chromadb
import numpy as np

from src.config import get_config, path

IST = timezone(timedelta(hours=5, minutes=30))
META_KEYS = ("source_url", "slug", "scheme_name", "source_type", "chunk_type",
             "section", "fetched_at", "n_tokens")
BATCH = 500
SMOKE_QUERY = "What is the exit load of HDFC Mid Cap Fund?"


def _client() -> chromadb.ClientAPI:
    persist_dir = path(get_config()["vector_db"]["persist_dir"])
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


def get_collection(create: bool = False):
    """Return the collection. With create=False (query time) it must already exist."""
    cfg = get_config()
    name = cfg["vector_db"]["collection"]
    client = _client()
    if not create:
        try:
            return client.get_collection(name, embedding_function=None)
        except Exception as e:
            raise RuntimeError(f"Collection '{name}' not found - run `python -m src.pipeline index`.") from e
    return client.get_or_create_collection(
        name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine", "embedding_model": cfg["embedding"]["model_name"]},
    )


def collection_model(col) -> str | None:
    return (col.metadata or {}).get("embedding_model")


def query(vector: np.ndarray, k: int, where: dict | None = None, col=None) -> list[dict]:
    """Top-k search. Returns [{chunk_id, text, meta, similarity}], best first."""
    col = col or get_collection()
    r = col.query(query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()], n_results=k,
                  where=where or None, include=["documents", "metadatas", "distances"])
    return [{"chunk_id": i, "text": d, "meta": m, "similarity": round(1 - dist, 4)}   # cosine distance -> similarity
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


# ---------------------------------------------------------------- indexing
def _load_inputs():
    chunks_path = path("data/03_chunks/chunks.jsonl")
    emb_dir = path("data/04_embeddings")
    for p in (chunks_path, emb_dir / "embeddings.npy", emb_dir / "ids.json", emb_dir / "manifest.json"):
        if not p.exists():
            raise SystemExit(f"Missing {p.relative_to(path(''))} - run the chunk and embed steps first.")

    with open(chunks_path, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    vecs = np.load(emb_dir / "embeddings.npy")
    ids = json.loads((emb_dir / "ids.json").read_text(encoding="utf-8"))
    manifest = json.loads((emb_dir / "manifest.json").read_text(encoding="utf-8"))

    # Consistency checks between steps 4, 5 and config
    chunks_sha = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
    if manifest.get("chunks_sha256") != chunks_sha:
        raise SystemExit("Embeddings are out of date with chunks.jsonl - re-run `python -m src.pipeline embed`.")
    if ids != [c["chunk_id"] for c in chunks] or len(vecs) != len(chunks):
        raise SystemExit("embeddings.npy / ids.json do not line up with chunks.jsonl - re-run embed.")
    if manifest["model_name"] != get_config()["embedding"]["model_name"]:
        raise SystemExit(f"Embeddings were made with {manifest['model_name']}, but config.yaml says "
                         f"{get_config()['embedding']['model_name']} - re-run embed.")
    return chunks, vecs, manifest


def _metadata(chunk: dict) -> dict:
    # Chroma accepts only str/int/float/bool values - never None.
    meta = {}
    for k in META_KEYS:
        v = chunk.get(k)
        meta[k] = v if isinstance(v, (int, float, bool)) else ("" if v is None else str(v))
    return meta


def run() -> dict:
    chunks, vecs, manifest = _load_inputs()
    cfg = get_config()
    client = _client()
    name = cfg["vector_db"]["collection"]

    # If the existing collection was built with another model or dimension, rebuild it.
    existing = {c.name if hasattr(c, "name") else c for c in client.list_collections()}
    if name in existing:
        old = client.get_collection(name, embedding_function=None)
        old_dim = None
        if old.count():
            sample = old.get(limit=1, include=["embeddings"])["embeddings"]
            old_dim = len(sample[0]) if sample is not None and len(sample) else None
        if collection_model(old) != manifest["model_name"] or (old_dim and old_dim != manifest["dim"]):
            print(f"  Rebuilding '{name}' (was model={collection_model(old)}, dim={old_dim})")
            client.delete_collection(name)

    col = get_collection(create=True)
    for i in range(0, len(chunks), BATCH):
        part = chunks[i:i + BATCH]
        col.upsert(
            ids=[c["chunk_id"] for c in part],
            embeddings=vecs[i:i + BATCH].tolist(),
            documents=[c["text"] for c in part],
            metadatas=[_metadata(c) for c in part],
        )

    # Remove chunks that no longer exist (e.g. a source was dropped or re-chunked)
    keep = {c["chunk_id"] for c in chunks}
    stale = [i for i in col.get(include=[])["ids"] if i not in keep]
    if stale:
        col.delete(ids=stale)

    count = col.count()
    if count != len(chunks):
        raise RuntimeError(f"Collection has {count} items but there are {len(chunks)} chunks")

    # Smoke test: the Mid Cap exit-load fact should be in the top 3
    from src.embedder import get_embedder
    hits = query(get_embedder().encode_query(SMOKE_QUERY), k=3, col=col)
    smoke_ok = any(h["meta"]["slug"] == "hdfc-mid-cap" and h["meta"]["section"] == "exit_load" for h in hits)

    report = {
        "collection": name,
        "persist_dir": cfg["vector_db"]["persist_dir"],
        "distance": (col.metadata or {}).get("hnsw:space"),
        "embedding_model": collection_model(col),
        "dim": manifest["dim"],
        "count": count,
        "upserted": len(chunks),
        "stale_removed": len(stale),
        "smoke_test": {
            "query": SMOKE_QUERY,
            "passed": smoke_ok,
            "top3": [{"chunk_id": h["chunk_id"], "similarity": h["similarity"],
                      "section": h["meta"]["section"], "text": h["text"][:120]} for h in hits],
        },
        "indexed_at": datetime.now(IST).isoformat(timespec="seconds"),
    }
    with open(path("data/05_vectordb/index_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"  Collection '{name}': {count} items (removed {len(stale)} stale)")
    print(f"  Smoke test {'PASSED' if smoke_ok else 'FAILED'}: '{SMOKE_QUERY}'")
    for h in hits:
        print(f"    {h['similarity']:.3f}  {h['text'][:90]}")
    return report


if __name__ == "__main__":
    run()
