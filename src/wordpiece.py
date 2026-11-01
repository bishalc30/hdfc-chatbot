"""Pure-Python BERT WordPiece tokenizer (no `tokenizers` / Hugging Face dependency).

Implements exactly what all-MiniLM-L6-v2's tokenizer.json specifies:
  BertNormalizer(clean_text, handle_chinese_chars, lowercase, strip_accents follows lowercase)
  -> BertPreTokenizer (split on whitespace and punctuation)
  -> WordPiece (unk "[UNK]", prefix "##", max 100 chars per word)
  -> "[CLS] ... [SEP]", truncated to max_length.
Used on Vercel to keep the function small; tests/test_wordpiece.py checks it gives the
same token ids as the `tokenizers` library.
"""
import json
import unicodedata


def _is_control(ch: str) -> bool:
    if ch in "\t\n\r":
        return False
    return unicodedata.category(ch).startswith("C")


def _is_whitespace(ch: str) -> bool:
    return ch in " \t\n\r" or unicodedata.category(ch) == "Zs"


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_chinese(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPieceTokenizer:
    def __init__(self, tokenizer_json: str, max_length: int = 256):
        with open(tokenizer_json, encoding="utf-8") as f:
            spec = json.load(f)
        model = spec["model"]
        self.vocab: dict[str, int] = model["vocab"]
        self.unk = model.get("unk_token", "[UNK]")
        self.prefix = model.get("continuing_subword_prefix", "##")
        self.max_chars = model.get("max_input_chars_per_word", 100)
        self.cls_id, self.sep_id = self.vocab["[CLS]"], self.vocab["[SEP]"]
        self.unk_id = self.vocab[self.unk]
        self.max_length = max_length

    # -- normaliser ------------------------------------------------------------
    @staticmethod
    def normalize(text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_chinese(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        text = "".join(out).lower()
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # -- pre-tokeniser -----------------------------------------------------------
    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        words = []
        for chunk in text.split():
            cur = []
            for ch in chunk:
                if _is_punctuation(ch):
                    if cur:
                        words.append("".join(cur))
                        cur = []
                    words.append(ch)
                else:
                    cur.append(ch)
            if cur:
                words.append("".join(cur))
        return words

    # -- wordpiece ---------------------------------------------------------------
    def wordpiece(self, word: str) -> list[int]:
        if len(word) > self.max_chars:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, found = len(word), None
            while start < end:
                piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            ids.append(found)
            start = end
        return ids

    def encode(self, text: str) -> list[int]:
        ids = []
        for word in self.pre_tokenize(self.normalize(text)):
            ids.extend(self.wordpiece(word))
        ids = ids[: self.max_length - 2]
        return [self.cls_id] + ids + [self.sep_id]
