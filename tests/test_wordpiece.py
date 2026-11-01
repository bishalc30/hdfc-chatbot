"""The pure-Python tokenizer (used on Vercel) must match the `tokenizers` library exactly.

Needs the exported model files (python -m src.pipeline export); skipped otherwise.
Run: python -m pytest tests/test_wordpiece.py -q
"""
import json

import pytest

from src.config import path

TOK = path("models/all-MiniLM-L6-v2-onnx/tokenizer.json")
pytestmark = pytest.mark.skipif(not TOK.exists(), reason="run `python -m src.pipeline export` first")

TRICKY = [
    "What is the exit load of HDFC Mid Cap Fund?",
    "do you advice to invedst on this ?",
    "Min. SIP ₹100 — exit load 1% (if redeemed within 1 year).",
    "Café résumé naïve ÀÉÎÕÜ",
    "HDFC 中文 test 日本",
    "e-mail: a.b@x.com, url https://groww.in/mutual-funds/hdfc-elss",
    "NIFTY Midcap 150 Total Return Index",
    "tabs\tnew\nlines\r\nand   spaces",
    "supercalifragilisticexpialidocious" * 4,
    "",
    "!!!???...",
    "word " * 400,                                     # longer than 256 tokens -> truncation
]


def _chunk_texts() -> list[str]:
    p = path("serving/chunks.json")
    return json.loads(p.read_text(encoding="utf-8"))["documents"] if p.exists() else []


@pytest.fixture(scope="module")
def tokenizers_pair():
    from tokenizers import Tokenizer

    from src.wordpiece import WordPieceTokenizer
    ref = Tokenizer.from_file(str(TOK))
    ref.enable_truncation(max_length=256)
    ref.no_padding()
    return ref, WordPieceTokenizer(str(TOK), max_length=256)


@pytest.mark.parametrize("text", TRICKY + _chunk_texts())
def test_same_ids_as_tokenizers_library(tokenizers_pair, text):
    ref, ours = tokenizers_pair
    assert ours.encode(text) == ref.encode(text).ids


def test_onnx_embedder_matches_index():
    import numpy as np

    from src.onnx_embedder import OnnxEmbedder
    chunks = json.loads(path("serving/chunks.json").read_text(encoding="utf-8"))
    vecs = np.load(path("serving/vectors.npy"))
    got = OnnxEmbedder().encode(chunks["documents"])
    assert float((got * vecs).sum(axis=1).min()) > 0.9999
