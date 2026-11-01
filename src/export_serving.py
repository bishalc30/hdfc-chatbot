"""Step 6b - Export a compact serving index for Vercel.

Reads the ChromaDB collection (the vector DB built in step 6) and writes:
  serving/vectors.npy     float32 (n, 384), one row per chunk
  serving/chunks.json     ids, documents, metadatas (same row order)
  serving/links.json      scheme -> page / factsheet links, AMC page
  serving/manifest.json   model, dim, count, source collection, ONNX check result
and downloads the ONNX model files to models/all-MiniLM-L6-v2-onnx/.

It then checks that the ONNX model gives the same vectors as sentence-transformers,
so queries on Vercel are embedded exactly like the indexed chunks.
Both serving/ and models/ are git-ignored; they are uploaded only by `vercel deploy`.
"""
import glob
import json
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np

from src.config import get_config, path
from src.onnx_embedder import MODEL_DIR

IST = timezone(timedelta(hours=5, minutes=30))
HF_REPO = "sentence-transformers/all-MiniLM-L6-v2"
MIN_COSINE = 0.999   # ONNX vs sentence-transformers agreement required


def download_onnx_model() -> None:
    from huggingface_hub import hf_hub_download
    out = path(MODEL_DIR)
    out.mkdir(parents=True, exist_ok=True)
    for remote, local in (("onnx/model.onnx", "model.onnx"), ("tokenizer.json", "tokenizer.json")):
        if not (out / local).exists():
            shutil.copy(hf_hub_download(HF_REPO, remote), out / local)
    print(f"  ONNX model ready in {MODEL_DIR}/")


def source_links() -> dict:
    links = {"schemes": {}, "amc": None}
    for p in glob.glob(str(path("data/02_cleaned/*.json"))):
        if p.endswith("load_report.json"):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d["source_type"] == "scheme":
            links["schemes"][d["scheme_name"]] = {
                "page": d["url"], "factsheet": d.get("official_links", {}).get("sid") or d["url"]}
        elif d["source_type"] == "amc_overview":
            links["amc"] = d["url"]
    return links


def run() -> dict:
    from src.vector_store import collection_model, get_collection

    cfg = get_config()
    col = get_collection()
    model_name = collection_model(col)
    if model_name != HF_REPO:
        raise SystemExit(f"Index was built with '{model_name}'; the ONNX export supports '{HF_REPO}' only.")

    data = col.get(include=["embeddings", "documents", "metadatas"])
    ids, docs, metas = data["ids"], data["documents"], data["metadatas"]
    vecs = np.asarray(data["embeddings"], dtype=np.float32)
    if len(ids) == 0:
        raise SystemExit("The collection is empty - run `python -m src.pipeline index` first.")

    # Same model in ONNX form; prove it embeds like sentence-transformers did.
    download_onnx_model()
    from src.onnx_embedder import OnnxEmbedder
    onnx = OnnxEmbedder()
    sample = list(range(0, len(docs), max(1, len(docs) // 10)))[:10]
    onnx_vecs = onnx.encode([docs[i] for i in sample])
    cosines = (onnx_vecs * vecs[sample]).sum(axis=1)
    if cosines.min() < MIN_COSINE:
        raise SystemExit(f"ONNX vectors differ from the index (min cosine {cosines.min():.6f}).")
    print(f"  ONNX check passed on {len(sample)} chunks (min cosine {cosines.min():.6f})")

    out = path("serving")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "vectors.npy", vecs)
    with open(out / "chunks.json", "w", encoding="utf-8") as f:
        json.dump({"ids": ids, "documents": docs, "metadatas": metas}, f, ensure_ascii=False)
    with open(out / "links.json", "w", encoding="utf-8") as f:
        json.dump(source_links(), f, indent=2, ensure_ascii=False)
    manifest = {
        "embedding_model": model_name,
        "dim": int(vecs.shape[1]),
        "count": len(ids),
        "source_collection": cfg["vector_db"]["collection"],
        "onnx_min_cosine": round(float(cosines.min()), 6),
        "exported_at": datetime.now(IST).isoformat(timespec="seconds"),
    }
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Exported {len(ids)} chunks x {vecs.shape[1]} dims to serving/")
    return manifest


if __name__ == "__main__":
    run()
