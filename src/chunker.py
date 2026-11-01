"""Step 4 - Chunking.

Reads data/02_cleaned/<slug>.json and writes:
  data/03_chunks/chunks.jsonl   one chunk per line (text + metadata)
  data/03_chunks/stats.json     counts and token-length summary

Strategy (based on the data):
  fact    - one chunk per field, prefixed with the scheme/AMC name, so each fact
            can be retrieved and cited on its own.
  summary - one overview chunk per scheme (category, plan, riskometer, benchmark, key costs)
            for broad questions like "tell me about HDFC Mid Cap".
  section - page text split at headings, then packed by whole sentences up to
            max_tokens with overlap_tokens of overlap.
Every chunk stays within the 256-token input limit of all-MiniLM-L6-v2.
"""
import glob
import hashlib
import json
import re
from functools import lru_cache
from statistics import mean

from transformers import AutoTokenizer

from src.config import get_config, path

MODEL_MAX_TOKENS = 256  # all-MiniLM-L6-v2 truncates input beyond this

LABELS = {
    # scheme fields
    "scheme_full_name": "Full scheme name",
    "fund_house": "Fund house (AMC)",
    "category": "Category",
    "plan": "Plan and option",
    "expense_ratio": "Expense ratio",
    "exit_load": "Exit load",
    "min_sip": "Minimum SIP amount",
    "min_lumpsum": "Minimum lumpsum (first) investment",
    "min_additional_investment": "Minimum additional investment",
    "lock_in": "Lock-in period",
    "riskometer": "Riskometer (risk level)",
    "benchmark": "Benchmark index",
    "fund_size_aum": "Fund size (AUM)",
    "fund_manager": "Fund manager(s)",
    "launch_date": "Launch date",
    "registrar": "Registrar and transfer agent (RTA)",
    "stamp_duty": "Stamp duty",
    # AMC fields
    "incorporation_date": "Incorporation date",
    "sponsor": "Sponsor",
    "trustee_organisation": "Trustee organisation",
    "ceo": "Chief Executive Officer",
    "managing_director": "Managing Director",
    "chief_investment_officer": "Chief Investment Officer",
    "total_aum": "Total assets under management (AUM)",
}


@lru_cache
def tokenizer():
    return AutoTokenizer.from_pretrained(get_config()["embedding"]["model_name"])


def n_tokens(text: str) -> int:
    """Token count including [CLS]/[SEP], as the embedding model sees it."""
    return len(tokenizer().encode(text, add_special_tokens=True))


def chunk_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:16]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def split_long_sentence(sentence: str, budget: int) -> list[str]:
    """Fallback for a single sentence longer than the budget: split on words."""
    pieces, cur = [], []
    for word in sentence.split():
        if cur and n_tokens(" ".join(cur + [word])) > budget:
            pieces.append(" ".join(cur))
            cur = []
        cur.append(word)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def split_text(prefix: str, text: str, max_tokens: int, overlap: int) -> list[str]:
    """Pack whole sentences up to max_tokens (prefix included), with sentence-level overlap."""
    budget = max_tokens - n_tokens(prefix)
    units = []
    for s in sentences(text):
        units += split_long_sentence(s, budget) if n_tokens(s) > budget else [s]

    out, cur = [], []
    for unit in units:
        if cur and n_tokens(prefix + " ".join(cur + [unit])) > max_tokens:
            out.append(" ".join(cur))
            tail, used = [], 0                      # carry the last sentences as overlap
            for prev in reversed(cur):
                used += n_tokens(prev)
                if used > overlap:
                    break
                tail.insert(0, prev)
            cur = tail
            while cur and n_tokens(prefix + " ".join(cur + [unit])) > max_tokens:
                cur.pop(0)                          # overlap must never push a chunk over the limit
        cur.append(unit)
    if cur:
        out.append(" ".join(cur))
    return [prefix + piece for piece in out]


def display_name(doc: dict) -> str:
    f = doc.get("fields", {})
    if doc["source_type"] == "scheme":
        return f.get("scheme_full_name") or doc.get("scheme_name") or doc["slug"]
    if doc["source_type"] == "amc_overview":
        return f.get("fund_house") or "HDFC Mutual Fund"
    return doc.get("scheme_name") or doc["slug"]


def summary_text(name: str, f: dict) -> str | None:
    if not f:
        return None
    parts = [f"{name} is a {f.get('category', 'mutual fund')} scheme"
             + (f" of {f['fund_house']}" if f.get("fund_house") else "")
             + (f" ({f['plan']})" if f.get("plan") else "") + "."]
    for key in ("riskometer", "benchmark", "expense_ratio", "exit_load", "min_sip", "lock_in"):
        if f.get(key):
            parts.append(f"{LABELS[key]}: {f[key]}.")
    return " ".join(parts)


def chunk_document(doc: dict, max_tokens: int, overlap: int) -> list[dict]:
    name = display_name(doc)
    base = {
        "source_url": doc["url"],
        "slug": doc["slug"],
        "scheme_name": doc.get("scheme_name") or "",
        "source_type": doc["source_type"],
        "fetched_at": doc["fetched_at"],
    }
    chunks = []

    for key, value in doc.get("fields", {}).items():
        text = f"{name} — {LABELS.get(key, key.replace('_', ' ').capitalize())}: {value}"
        text = text if text.endswith(".") else text + "."
        chunks.append({**base, "chunk_type": "fact", "section": key, "text": text,
                       "chunk_id": chunk_id(doc["slug"], "fact", key)})

    if doc["source_type"] == "scheme":
        text = summary_text(name, doc.get("fields", {}))
        if text:
            chunks.append({**base, "chunk_type": "summary", "section": "summary", "text": text,
                           "chunk_id": chunk_id(doc["slug"], "summary")})

    for s_idx, sec in enumerate(doc.get("sections", [])):
        heading = sec["heading"].strip() or "Overview"
        prefix = f"{name} — {heading}: "
        for i, piece in enumerate(split_text(prefix, sec["text"], max_tokens, overlap)):
            chunks.append({**base, "chunk_type": "section", "section": heading[:100], "text": piece,
                           "chunk_id": chunk_id(doc["slug"], "section", s_idx, heading, i)})

    for c in chunks:
        c["n_tokens"] = n_tokens(c["text"])
        if c["n_tokens"] > MODEL_MAX_TOKENS:
            raise ValueError(f"Chunk {c['chunk_id']} has {c['n_tokens']} tokens (> {MODEL_MAX_TOKENS})")
    return chunks


def run() -> list[dict]:
    cfg = get_config()["chunking"]
    max_tokens, overlap = cfg["max_tokens"], cfg["overlap_tokens"]
    if max_tokens > MODEL_MAX_TOKENS:
        raise ValueError(f"chunking.max_tokens ({max_tokens}) exceeds the model limit ({MODEL_MAX_TOKENS})")

    doc_paths = sorted(p for p in glob.glob(str(path("data/02_cleaned/*.json")))
                       if not p.endswith("load_report.json"))
    if not doc_paths:
        raise SystemExit("No cleaned documents in data/02_cleaned - run `python -m src.pipeline load` first.")

    all_chunks = []
    for p in doc_paths:
        with open(p, encoding="utf-8") as f:
            doc = json.load(f)
        chunks = chunk_document(doc, max_tokens, overlap)
        all_chunks += chunks
        print(f"  {doc['slug']:<16} {len(chunks):>3} chunks")

    ids = [c["chunk_id"] for c in all_chunks]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"Duplicate chunk_ids: {sorted(dupes)}")

    out_dir = path("data/03_chunks")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    tokens = [c["n_tokens"] for c in all_chunks]
    count_by = lambda key: {v: sum(1 for c in all_chunks if c[key] == v) for v in sorted({c[key] for c in all_chunks})}
    buckets = [(0, 32), (33, 64), (65, 128), (129, 192), (193, 256)]
    stats = {
        "total_chunks": len(all_chunks),
        "by_chunk_type": count_by("chunk_type"),
        "by_source": count_by("slug"),
        "tokens": {"min": min(tokens), "max": max(tokens), "mean": round(mean(tokens), 1)},
        "token_histogram": {f"{lo}-{hi}": sum(lo <= t <= hi for t in tokens) for lo, hi in buckets},
        "settings": {"max_tokens": max_tokens, "overlap_tokens": overlap,
                     "model_limit": MODEL_MAX_TOKENS},
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Done: {len(all_chunks)} chunks, tokens min/mean/max = "
          f"{stats['tokens']['min']}/{stats['tokens']['mean']}/{stats['tokens']['max']}")
    return all_chunks


if __name__ == "__main__":
    run()
