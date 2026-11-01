"""Step 7 - Retrieval.

Embeds the question with the SAME model used at ingestion, searches ChromaDB, filters by
scheme when the question names one, and applies the similarity threshold.
"""
import re
from functools import lru_cache

from src.config import get_config
from src.embedder import get_embedder
from src.vector_store import collection_model, get_collection, query

# Aliases -> scheme_name exactly as stored in metadata (from sources.csv)
SCHEME_ALIASES = {
    r"\blarge\s*-?\s*cap\b":                        "HDFC Large Cap Fund",
    r"\bflexi\s*-?\s*cap\b|\bequity fund\b":       "HDFC Flexi Cap Fund",
    r"\belss\b|\btax\s*saver\b|\btax\s*saving\b":  "HDFC ELSS Tax Saver Fund",
    r"\bmid\s*-?\s*cap\b":                          "HDFC Mid Cap Fund",
}


def detect_schemes(question: str) -> list[str]:
    return [name for pat, name in SCHEME_ALIASES.items() if re.search(pat, question, re.I)]


class Retriever:
    def __init__(self):
        cfg = get_config()
        self.top_k = cfg["retrieval"]["top_k"]
        self.threshold = cfg["retrieval"]["similarity_threshold"]
        self.embedder = get_embedder()
        self.col = get_collection()
        stored = collection_model(self.col)
        if stored != self.embedder.model_name:
            raise RuntimeError(f"Embedding model mismatch: index uses '{stored}', config uses "
                               f"'{self.embedder.model_name}'. Re-run `python -m src.pipeline all`.")

    def search(self, question: str) -> dict:
        schemes = detect_schemes(question)
        vec = self.embedder.encode_query(question)

        where = None
        if len(schemes) == 1:
            where = {"scheme_name": schemes[0]}
        elif len(schemes) > 1:
            where = {"scheme_name": {"$in": schemes}}

        hits = query(vec, self.top_k, where, col=self.col) if where else []
        if len(hits) < self.top_k:                          # top up with general chunks (AMC, how-to)
            seen = {h["chunk_id"] for h in hits}
            hits += [h for h in query(vec, self.top_k, None, col=self.col) if h["chunk_id"] not in seen]
        hits = sorted(hits, key=lambda h: h["similarity"], reverse=True)[: self.top_k]

        return {
            "schemes_detected": schemes,
            "filter": where,
            "hits": hits,
            "confident": bool(hits) and hits[0]["similarity"] >= self.threshold,
            "threshold": self.threshold,
        }


@lru_cache
def get_retriever() -> Retriever:
    return Retriever()
