"""ONNX version of sentence-transformers/all-MiniLM-L6-v2 for serving (no PyTorch).

Reproduces SentenceTransformer.encode(..., normalize_embeddings=True):
  tokenize (max 256 tokens) -> transformer -> mean pooling over real tokens -> L2 normalise.
Tokenisation uses the pure-Python WordPiece tokenizer in src/wordpiece.py (no `tokenizers`).
The export step (src/export_serving.py) checks that these vectors match the
sentence-transformers vectors used to build the index.

Files (downloaded by the export step, git-ignored):
  models/all-MiniLM-L6-v2-onnx/model.onnx
  models/all-MiniLM-L6-v2-onnx/tokenizer.json
"""
from functools import lru_cache

import numpy as np

from src.config import get_config, path

MODEL_DIR = "models/all-MiniLM-L6-v2-onnx"
MAX_SEQ_LENGTH = 256  # same as SentenceTransformer("all-MiniLM-L6-v2").max_seq_length


class OnnxEmbedder:
    def __init__(self, model_dir: str = MODEL_DIR):
        import onnxruntime as ort

        from src.wordpiece import WordPieceTokenizer

        d = path(model_dir)
        for f in ("model.onnx", "tokenizer.json"):
            if not (d / f).exists():
                raise RuntimeError(f"Missing {model_dir}/{f} - run `python -m src.pipeline export`.")
        self.model_name = get_config()["embedding"]["model_name"]
        self.tokenizer = WordPieceTokenizer(str(d / "tokenizer.json"), max_length=MAX_SEQ_LENGTH)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(d / "model.onnx"), sess_options=opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = self.session.get_outputs()[0].shape[-1]

    def encode(self, texts: list[str]) -> np.ndarray:
        seqs = [self.tokenizer.encode(t) for t in texts]
        width = max(len(s) for s in seqs)
        ids = np.zeros((len(seqs), width), dtype=np.int64)           # [PAD] id is 0
        mask = np.zeros((len(seqs), width), dtype=np.int64)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = s
            mask[i, : len(s)] = 1
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]                     # (batch, tokens, 384)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return (pooled / np.clip(norms, 1e-12, None)).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache
def get_onnx_embedder() -> OnnxEmbedder:
    return OnnxEmbedder()
