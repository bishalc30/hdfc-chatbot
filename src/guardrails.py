"""Guardrails used by the answer flow (step 7) and tested in step 9.

  find_pii(text)       -> PII *types* found (never the values)
  classify(text)       -> "pii" | "advice" | "performance" | "factual"
  validate_answer(...) -> checks the LLM output: citation, length, leaks, NOT_IN_SOURCES
"""
import re
from dataclasses import dataclass

# ---------------------------------------------------------------- input checks
PII_PATTERNS = {
    "pan":     re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.I),
    "aadhaar": re.compile(r"(?<!\d)\d{4}[\s-]?\d{4}[\s-]?\d{4}(?!\d)"),
    "phone":   re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)"),
    "email":   re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "otp":     re.compile(r"\b(?:otp|one[\s-]?time\s+pass(?:word|code))\b\D{0,20}\d{4,8}\b", re.I),
    "account": re.compile(r"(?<!\d)\d{9,18}(?!\d)"),
}

# --- normalisation: lower-case, casual spellings, and typo correction --------------------
# Casual / SMS spellings mapped to their standard form before matching.
CASUAL = {
    "shud": "should", "shld": "should", "shuld": "should", "shoud": "should", "sould": "should",
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "gud": "good",
    "gd": "good", "invst": "invest", "reco": "recommend", "rec": "recommend", "wrth": "worth",
    "shouldnt": "should not", "isnt": "is not", "wont": "will not", "dont": "do not",
    "worht": "worth", "wroth": "worth", "advse": "advise", "advic": "advice",
}
# Real words that sit close to an advice word but must never be "corrected".
KEEP = {"risks", "goods", "ideas", "select", "proper", "safety", "holder", "holders", "shares",
        "sector", "seller", "buyer", "better", "invested", "investor", "investors"}
# Words that signal advice. Tokens (length >= 5, same first letter) within a small edit distance
# of these are corrected, so "invedst", "recomend", "advce", "suggst" still match.
ADVICE_VOCAB = [
    "advice", "advise", "advisable", "advisor", "recommend", "recommendation", "suggest",
    "suggestion", "opinion", "prefer", "invest", "investing", "investment", "should", "worth",
    "better", "best", "good", "buy", "buying", "sell", "selling", "hold", "holding", "redeem",
    "switch", "portfolio", "allocate", "allocation", "profitable", "safe", "risky", "wise",
    "choose", "suitable", "ideal",
]
_VOCAB_BY_LETTER: dict[str, list[str]] = {}
for _w in ADVICE_VOCAB:
    _VOCAB_BY_LETTER.setdefault(_w[0], []).append(_w)


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Damerau-Levenshtein (optimal string alignment) distance, capped at limit + 1.
    A swapped pair of letters ("invset") counts as one edit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
        if min(d[i]) > limit:
            return limit + 1
    return d[-1][-1]


def _correct(token: str) -> str:
    if token in CASUAL:
        return CASUAL[token]
    if len(token) < 5 or not token.isalpha() or token in ADVICE_VOCAB or token in KEEP:
        return token
    limit = 1 if len(token) <= 6 else 2
    best, best_d = token, limit + 1
    for word in _VOCAB_BY_LETTER.get(token[0], []):
        d = _edit_distance(token, word, limit)
        if d < best_d:
            best, best_d = word, d
    return best


def normalize(text: str) -> str:
    """Lower-case, strip punctuation, expand casual spellings, fix typos of advice words."""
    text = text.lower().replace("’", "'").replace("'", "")
    tokens = re.findall(r"[a-z]+|\d+(?:\.\d+)?%?|₹", text)
    return " ".join(_correct(t) for t in tokens)


# --- advice / performance patterns (matched on the normalised text) ---------------------
ACTION = r"(invest|investing|buy|buying|sell|selling|hold|holding|redeem|exit|switch|put|start|stop|continue|choose|pick|go)"
ADVICE_PATTERNS = [
    r"\b(advice|advise|advisable|advisor|recommend\w*|suggest\w*|opinion|prefer\w*)\b",
    # "should i ...", "should i?" - but not procedural questions like "should i pay exit load ..."
    r"\b(should|shall|must|ought)( not)? (i|we|one|me)\b"
    r"(?!.{0,30}\b(pay|know|expect|submit|fill|provide|keep|note|check|download|update|complete|do kyc)\b)",
    r"\b(would|will) (it|this|that) be (good|wise|smart|safe|right|better|worth)\b",
    r"\b(is|are|was|would|will)\b.{0,40}\b(good|bad|safe|risky|wise|right|smart|profitable|"
    r"better|best|ideal|suitable)\b.{0,20}\b(to )?" + ACTION + r"?",
    r"\bworth\b",                                                          # "worth it", "worth buying"
    r"\b(good|bad|right|best|ideal|safe) (time|fund|scheme|option|choice|investment|pick|bet)\b",
    r"\b(better|best)\b",
    r"\bwhich\b.{0,40}\b(choose|pick|go for|invest|buy|should|prefer)\b",
    r"\b(buy|sell|hold|exit|redeem|switch)\b (now|or|today|yet|it|this|that)\b",
    r"\b(invest|buy)\b (in|on|into) (this|it|that)\b",
    r"\bput (my |our )?(money|savings)\b",
    r"\bgo for (it|this|that)\b",
    r"\b(my|our) (portfolio|savings|retirement|goals?)\b",
    r"\b(allocate|allocation)\b",
    r"\b(can|could) you (tell|help) me (if|whether|what) to\b",
    r"\bignore (all|the|your|previous|prior) (rules|instructions)\b",      # prompt injection
]
ADVICE = re.compile("|".join(f"(?:{p})" for p in ADVICE_PATTERNS))
PERFORMANCE = re.compile(
    r"\b(returns?|cagr|xirr|performance|performed|how much .* (earn|grow|make)|nav history|"
    r"beat the (market|index))\b")


def find_pii(text: str) -> list[str]:
    """Return the PII types present. Values are never returned or logged."""
    return [name for name, pat in PII_PATTERNS.items() if pat.search(text)]


def is_advice(text: str) -> bool:
    return bool(ADVICE.search(normalize(text)))


def classify(text: str) -> str:
    """Order matters: PII (on raw text) -> advice -> performance -> factual."""
    if find_pii(text):
        return "pii"
    norm = normalize(text)
    if ADVICE.search(norm):
        return "advice"
    if PERFORMANCE.search(norm):
        return "performance"
    return "factual"


# ---------------------------------------------------------------- output checks
SOURCE_LINE = re.compile(r"^\s*\**SOURCE\**\s*:\s*<?(\S+?)>?\s*$", re.I | re.M)
URL = re.compile(r"https?://\S+")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z₹(\"'])")
OUTPUT_LEAK = re.compile(
    r"\b(you should|i (would )?recommend|i suggest|better (option|choice)|good investment|"
    r"worth investing|consider investing)\b|\d+(\.\d+)?\s?%\s*(returns?|cagr|p\.?\s?a\.?)", re.I)


@dataclass
class ValidatedAnswer:
    ok: bool
    text: str = ""
    url: str | None = None
    reason: str = "ok"


def validate_answer(raw: str, allowed_urls: list[str], max_sentences: int) -> ValidatedAnswer:
    if not raw or "NOT_IN_SOURCES" in raw:
        return ValidatedAnswer(False, reason="not_in_sources")

    m = SOURCE_LINE.search(raw)
    url = m.group(1).rstrip(".,)") if m else None
    body = SOURCE_LINE.sub("", raw)
    body = URL.sub("", body)                                 # exactly one link: the citation
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return ValidatedAnswer(False, reason="empty_answer")

    if OUTPUT_LEAK.search(body):
        return ValidatedAnswer(False, reason="advice_or_performance_leak")

    reason = "ok"
    if url not in allowed_urls:                              # citation must come from retrieved chunks
        url, reason = allowed_urls[0], "citation_replaced"

    sentences = [s.strip() for s in SENTENCE_SPLIT.split(body) if s.strip()]
    if len(sentences) > max_sentences:
        body = " ".join(sentences[:max_sentences])
        reason = "trimmed" if reason == "ok" else reason + "+trimmed"
    return ValidatedAnswer(True, body, url, reason)
