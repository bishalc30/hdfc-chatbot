"""Step 7 - Retrieval.

Embeds the question with the SAME model used at ingestion, searches the vector index,
filters by scheme when the question names one, and applies the similarity threshold.

Two backends share the same search logic (RETRIEVAL_BACKEND env var):
  chroma  (default, local)  - sentence-transformers + the ChromaDB collection
  serving (Vercel)          - ONNX version of the same model + serving/ export of that collection,
                              exact cosine search with numpy (no PyTorch, no chromadb package)
"""
import json
import os
import re
from functools import lru_cache

from src.config import get_config, path

# Aliases -> scheme_name exactly as stored in metadata (from sources.csv)
SCHEME_ALIASES = {
    r"\blarge\s*-?\s*cap\b":                        "HDFC Large Cap Fund",
    r"\bflexi\s*-?\s*cap\b|\bequity fund\b":       "HDFC Flexi Cap Fund",
    r"\belss\b|\btax\s*saver\b|\btax\s*saving\b":  "HDFC ELSS Tax Saver Fund",
    r"\bmid\s*-?\s*cap\b":                          "HDFC Mid Cap Fund",
}


def detect_schemes(question: str) -> list[str]:
    return [name for pat, name in SCHEME_ALIASES.items() if re.search(pat, question, re.I)]


class BaseRetriever:
    model_name: str

    def __init__(self):
        cfg = get_config()
        self.top_k = cfg["retrieval"]["top_k"]
        self.threshold = cfg["retrieval"]["similarity_threshold"]
        if self.model_name != cfg["embedding"]["model_name"]:
            raise RuntimeError(f"Embedding model mismatch: index uses '{self.model_name}', config uses "
                               f"'{cfg['embedding']['model_name']}'. Re-run `python -m src.pipeline all`.")

    # backend-specific
    def embed(self, text: str):
        raise NotImplementedError

    def query(self, vec, k: int, where: dict | None) -> list[dict]:
        raise NotImplementedError

    # shared logic
    def search(self, question: str) -> dict:
        schemes = detect_schemes(question)
        vec = self.embed(question)

        where = None
        if len(schemes) == 1:
            where = {"scheme_name": schemes[0]}
        elif len(schemes) > 1:
            where = {"scheme_name": {"$in": schemes}}

        hits = self.query(vec, self.top_k, where) if where else []
        if len(hits) < self.top_k:                          # top up with general chunks (AMC, how-to)
            seen = {h["chunk_id"] for h in hits}
            hits += [h for h in self.query(vec, self.top_k, None) if h["chunk_id"] not in seen]
        hits = sorted(hits, key=lambda h: h["similarity"], reverse=True)[: self.top_k]

        return {
            "schemes_detected": schemes,
            "filter": where,
            "hits": hits,
            "confident": bool(hits) and hits[0]["similarity"] >= self.threshold,
            "threshold": self.threshold,
        }


class ChromaRetriever(BaseRetriever):
    """Local: sentence-transformers + ChromaDB (the vector DB built by the pipeline)."""

    def __init__(self):
        from src.embedder import get_embedder
        from src.vector_store import collection_model, get_collection
        self.embedder = get_embedder()
        self.col = get_collection()
        self.model_name = collection_model(self.col)
        super().__init__()

    def embed(self, text: str):
        return self.embedder.encode_query(text)

    def query(self, vec, k, where):
        from src.vector_store import query
        return query(vec, k, where, col=self.col)


class ServingRetriever(BaseRetriever):
    """Vercel: ONNX embedder + serving/ export of the ChromaDB collection."""

    def __init__(self):
        import numpy as np
        from src.onnx_embedder import get_onnx_embedder

        d = path("serving")
        if not (d / "vectors.npy").exists():
            raise RuntimeError("Serving index not found - run `python -m src.pipeline export`.")
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.np = np
        self.vectors = np.load(d / "vectors.npy")
        self.ids, self.docs, self.metas = chunks["ids"], chunks["documents"], chunks["metadatas"]
        self.model_name = manifest["embedding_model"]
        self.embedder = get_onnx_embedder()
        super().__init__()

    def embed(self, text: str):
        return self.embedder.encode_query(text)

    @staticmethod
    def _matches(meta: dict, where: dict | None) -> bool:
        if not where:
            return True
        for key, cond in where.items():
            value = meta.get(key)
            if isinstance(cond, dict) and "$in" in cond:
                if value not in cond["$in"]:
                    return False
            elif value != cond:
                return False
        return True

    def query(self, vec, k, where):
        sims = self.vectors @ vec                              # unit vectors -> cosine similarity
        order = self.np.argsort(-sims)
        out = []
        for i in order:
            if self._matches(self.metas[i], where):
                out.append({"chunk_id": self.ids[i], "text": self.docs[i], "meta": self.metas[i],
                            "similarity": round(float(sims[i]), 4)})
                if len(out) == k:
                    break
        return out


@lru_cache
def get_retriever() -> BaseRetriever:
    backend = os.getenv("RETRIEVAL_BACKEND", "chroma").strip().lower()
    if backend == "serving":
        return ServingRetriever()
    if backend == "chroma":
        return ChromaRetriever()
    raise ValueError(f"RETRIEVAL_BACKEND must be 'chroma' or 'serving', got '{backend}'")
