"""Conversational, guided flow on top of the grounded answer flow (src/answerer.py).

  message + state {"fund": <scheme_name> | None}
    0. PII                -> answer() rejects it (nothing else sees the message)
    1. navigation         "HDFC" -> fund step | a fund name -> question step | "Change fund"
    2. comparison         2+ funds, or compare / vs -> one fund at a time + fund pills
    3. advice             -> facts only + question pills (or fund pills)
    4. small_talk         -> short LLM reply + "Which fund house ...?" + [HDFC]
    5. factual            -> answer()  (the ONLY route that retrieves; follow-ups use the selected fund)

The server keeps no session: the client sends `state` with every message and stores the
`state` returned. Grounding, retrieval and validation of factual answers are unchanged.

CLI:  python -m src.chat "Hi, how are you?"
"""
import re
import sys
import uuid
from datetime import datetime

from src import guardrails as G
from src.answerer import IST, answer, write_trace
from src.config import get_config
from src.llm import get_llm
from src.retriever import SCHEME_ALIASES, detect_schemes

# scheme_name as stored in metadata -> name shown on pills and used in pill questions
FUNDS = {
    "HDFC Large Cap Fund": "HDFC Large Cap Fund",
    "HDFC Flexi Cap Fund": "HDFC Flexi Cap Fund",
    "HDFC ELSS Tax Saver Fund": "HDFC ELSS Tax Saver",
    "HDFC Mid Cap Fund": "HDFC Mid Cap Fund",
}
ELSS = "HDFC ELSS Tax Saver Fund"

# question pills: (label, follow-up pattern, question template, ELSS only)
FIELDS = [
    ("Expense ratio", r"\bexpense\s*ratios?\b|\bter\b|\bexpenses?\b", "What is the expense ratio of {f}?", False),
    ("Exit load", r"\bexit\s*loads?\b", "What is the exit load of {f}?", False),
    ("Minimum SIP amount", r"\bsip\b|\bminimum\s+(investment|amount)\b", "What is the minimum SIP amount for {f}?", False),
    ("Riskometer", r"\brisk\s*-?\s*o?\s*meter\b|\brisk\s+(level|grade|category)\b", "What is the riskometer level of {f}?", False),
    ("Benchmark", r"\bbench\s*marks?\b", "What is the benchmark of {f}?", False),
    ("Lock-in period", r"\block\s*-?\s*in\b", "What is the lock-in period of {f}?", True),
]

MSG = {
    "ask_house": "Which fund house would you like to explore?",
    "ask_fund": "Which HDFC fund would you like to check?",
    "ask_fact": "What would you like to know about {f}?",
    "comparison": "I can't compare funds, but I can share facts for one fund at a time. "
                  "Which fund would you like to check?",
    "advice": "I share facts only, not investment advice.",
    "advice_fund": "I share facts only, not investment advice. Here are the facts I can share about {f}.",
    "advice_nofund": "I share facts only, not investment advice. Pick a fund to see its facts.",
    "small_talk_fallback": "Hi! I can share quick facts about HDFC Mutual Fund schemes, such as the expense "
                           "ratio, exit load or minimum SIP. I share facts only, not investment advice.",
}
LABEL = {"comparison": "Comparison — one fund at a time", "advice": "Advice question — facts only"}

SMALL_TALK_PROMPT = """You are the HDFC MF Facts Assistant, a friendly chatbot.
Reply to the user's small-talk message naturally, in 1-2 short sentences.
In the reply, say that you share quick facts about HDFC Mutual Fund schemes (for example expense ratio,
exit load, minimum SIP, lock-in, riskometer, benchmark), facts only and no investment advice.
Do not ask any question. Do not give any fund facts, numbers, opinions or URLs."""

# --- message patterns (on lower-cased text without punctuation) ------------------------------
SMALL_TALK_PHRASES = re.compile(
    r"\b(hi+|hello+|hey+|hiya|yo|namaste|greetings|good (morning|afternoon|evening|day)|"
    r"how (are|r) (you|u)( doing)?( today)?|hows it going|how is it going|whats up|wassup|sup|"
    r"thanks?( (you|u|a lot))?|ok(ay)?|cool|great|nice|bye|goodbye|see (you|ya)|"
    r"who are (you|u)|what are (you|u)|what (can|do) (you|u) do|how (can|do|could) (you|u) help( me)?|"
    r"can (you|u) help( me)?|help( me)?|what is this|start( over)?|menu|i am (fine|good|well))\b")
FILLER = {"there", "bot", "assistant", "please", "so", "and", "i", "im", "am", "fine", "good", "well",
          "doing", "today", "me", "you", "u", "very", "much", "again", "too", "also", "all"}
HOUSE = re.compile(r"^(hdfc|hdfc (mf|amc|mutual funds?))$")
CHANGE_FUND = re.compile(r"^(change|switch|choose|pick|select|another|different|other) (the )?(fund|scheme)s?$")
COMPARE = re.compile(r"\b(compare|comparing|comparison|vs|versus|difference between|differences between)\b")
FUND_WORDS = re.compile(r"\b(hdfc|fund|funds|scheme|plan|direct|growth|the|select|selected|i|want|choose|pick)\b")
PRONOUN = re.compile(r"\b(this|that|the|same|selected) (fund|scheme)\b|\b(it|its)\b", re.I)
LEAD_IN = re.compile(r"^\s*(and|also|ok(ay)?|so|then)?\s*(what|how) about\s+|^\s*(and|also)\s+", re.I)


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower().replace("'", ""))).strip()


def display(fund: str | None) -> str | None:
    return FUNDS.get(fund) if fund else None


# --- pills -------------------------------------------------------------------------------------
def pill(label: str, message: str | None = None, kind: str = "option") -> dict:
    return {"label": label, "message": message or label, "kind": kind}


def house_pills() -> list[dict]:
    return [pill("HDFC")]


def fund_pills() -> list[dict]:
    return [pill(name) for name in FUNDS.values()]


def question_pills(fund: str) -> list[dict]:
    f = display(fund)
    out = [pill(label, tmpl.format(f=f)) for label, _, tmpl, elss in FIELDS if not elss or fund == ELSS]
    return out + [pill("Change fund", kind="secondary")]


def default_pills(fund: str | None) -> list[dict]:
    return question_pills(fund) if fund else fund_pills()


# --- routing helpers ---------------------------------------------------------------------------
def is_small_talk(p: str) -> bool:
    if not SMALL_TALK_PHRASES.search(p):
        return False
    leftover = SMALL_TALK_PHRASES.sub(" ", p).split()
    return all(w in FILLER for w in leftover)


def is_fund_selection(p: str) -> bool:
    """The message is just a fund name ("HDFC Mid Cap Fund", "mid cap"), not a question."""
    for pat in SCHEME_ALIASES:
        p = re.sub(pat, " ", p, flags=re.I)
    return not FUND_WORDS.sub(" ", p).split()


def field_of(text: str) -> tuple | None:
    return next((f for f in FIELDS if re.search(f[1], text, re.I)), None)


def contextualise(q: str, fund: str) -> str | None:
    """Turn a short follow-up ("what about exit load?") into a question about the selected fund."""
    f = display(fund)
    field = field_of(q)
    if field:
        return field[2].format(f=f)
    if PRONOUN.search(q):                                    # "who manages it?"
        return PRONOUN.sub(f, q, count=1)
    if LEAD_IN.match(q):                                     # "what about fund size?"
        rest = LEAD_IN.sub("", q).rstrip(" ?.!")
        return f"What is the {rest} of {f}?" if rest else None
    return None                                              # e.g. AMC questions: leave unchanged


def small_talk_reply(message: str) -> str:
    """LLM reply, 1-2 sentences, no questions / URLs / numbers; canned text if anything is off."""
    try:
        raw = get_llm().complete([{"role": "system", "content": SMALL_TALK_PROMPT},
                                  {"role": "user", "content": message}],
                                 temperature=0.4, max_tokens=get_config()["generation"]["max_tokens"])
    except Exception:
        return MSG["small_talk_fallback"]
    text = re.sub(r"\s+", " ", G.URL.sub("", raw or "")).strip().strip('"')
    sentences = [s.strip() for s in G.SENTENCE_SPLIT.split(text) if s.strip() and not s.strip().endswith("?")]
    text = " ".join(sentences[:2])
    if not text or len(text) > 320 or re.search(r"\d", text) or G.OUTPUT_LEAK.search(text):
        return MSG["small_talk_fallback"]
    return text


# --- main entry --------------------------------------------------------------------------------
def reply(route: str, text: str, pills: list[dict], fund: str | None, trace: dict | None = None,
          **extra) -> dict:
    r = {"route": route, "intent": route, "text": text, "url": None, "last_updated": None,
         "label": LABEL.get(route), "pills": pills, "state": {"fund": fund}, "trace": trace or {}}
    r.update(extra)
    return r


def chat(message: str, state: dict | None = None) -> dict:
    fund = (state or {}).get("fund")
    fund = fund if fund in FUNDS else None
    q = re.sub(r"\s+", " ", message or "").strip()[: get_config()["guardrails"]["max_input_chars"]]
    trace = {"trace_id": uuid.uuid4().hex[:8], "at": datetime.now(IST).isoformat(timespec="seconds")}

    def routed(route: str, text: str, pills: list[dict], new_fund: str | None, **extra) -> dict:
        write_trace(trace | {"event": "chat_route", "route": route, "question": q,
                             "selected_fund": new_fund, "response": {"intent": route, "text": text}})
        return reply(route, text, pills, new_fund, trace | {"route": route}, **extra)

    # 0. empty / PII: the answer flow handles both; PII never reaches any other step
    if not q or G.find_pii(q):
        r = answer(q)
        return reply("factual", r["text"], default_pills(fund) if fund else [], fund, r["trace"],
                     intent=r["intent"], url=r["url"], last_updated=r["last_updated"])

    p = plain(q)
    schemes = detect_schemes(q)

    # 1. navigation
    if CHANGE_FUND.match(p) or HOUSE.match(p):
        return routed("fund_step", MSG["ask_fund"], fund_pills(), fund)
    if len(schemes) == 1 and is_fund_selection(p):
        new = schemes[0]
        return routed("question_step", MSG["ask_fact"].format(f=display(new)), question_pills(new), new)

    # 2. comparison  3. advice
    if len(schemes) >= 2 or COMPARE.search(p):
        return routed("comparison", MSG["comparison"], fund_pills(), fund)
    if G.classify(q) == "advice":
        text = MSG["advice_fund"].format(f=display(fund)) if fund else MSG["advice_nofund"]
        return routed("advice", text, default_pills(fund), fund)

    # 4. small talk
    if is_small_talk(p):
        text = f"{small_talk_reply(q)}\n\n{MSG['ask_house']}"
        return routed("small_talk", text, house_pills(), fund)

    # 5. factual (incl. performance, which the answer flow refuses before retrieval)
    asked, context_fund = q, None
    if len(schemes) == 1:
        fund = schemes[0]                                    # named fund becomes the selected fund
    elif not schemes and fund:
        rewritten = contextualise(q, fund)
        if rewritten:
            asked, context_fund = rewritten, display(fund)
    r = answer(asked)
    if r["intent"] == "clarify" and not fund:
        pills = fund_pills()
    else:
        pills = question_pills(fund) if fund else []
    return reply("factual", r["text"], pills, fund, r["trace"], intent=r["intent"], url=r["url"],
                 last_updated=r["last_updated"], asked=asked if asked != q else None,
                 context_fund=context_fund)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.chat "your message" [selected fund]')
    out = chat(sys.argv[1], {"fund": sys.argv[2] if len(sys.argv) > 2 else None})
    print(f"[{out['route']} / {out['intent']}] fund={out['state']['fund']}")
    print(out["text"])
    if out["url"]:
        print("Source:", out["url"])
    print("Pills:", [x["label"] for x in out["pills"]])
