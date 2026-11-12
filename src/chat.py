"""Conversational, guided flow on top of the grounded answer flow (src/answerer.py).

Rule-based routing: only factual answers call the LLM (inside answer()), and repeated
(fund, fact) questions are served from a 24h cache without any LLM call.

  message + state {"fund": <scheme_name> | None} + optional pill {"action", "fund", "fact"}
    0. empty / PII        -> empty: answer() | PII: "please don't share personal details" (input never logged)
    1. pill click         -> its known action: fund step | question step | factual (fund, fact)
    2. typed navigation   "HDFC" / a fund name alone / "Change fund"
    3. other fund house   -> fallback (never answered as an HDFC fund)
    4. advice             -> fallback + AMFI investor-education link
    5. fact keyword       + fund named -> factual | + fund selected -> factual | neither -> "Which HDFC fund?"
       fund-house fact    (CEO, sponsor, total AUM, ...) -> factual
    6. closing / greeting -> fixed friendly replies
    7. anything else      -> fallback

The server keeps no session: the client sends `state` with every message and stores the
`state` returned. Grounding, retrieval and validation of factual answers are unchanged.

CLI:  python -m src.chat "What is the exit load of HDFC Mid Cap Fund?"
"""
import random
import re
import sys
import time
import uuid
from datetime import datetime
from difflib import SequenceMatcher
from functools import lru_cache

from src import guardrails as G
from src.answerer import IST, answer, write_trace
from src.config import get_config
from src.retriever import SCHEME_ALIASES, detect_schemes

# scheme_name as stored in metadata -> name shown on pills and used in questions
FUNDS = {
    "HDFC Large Cap Fund": "HDFC Large Cap Fund",
    "HDFC Flexi Cap Fund": "HDFC Flexi Cap Fund",
    "HDFC ELSS Tax Saver Fund": "HDFC ELSS Tax Saver",
    "HDFC Mid Cap Fund": "HDFC Mid Cap Fund",
}
ELSS = "HDFC ELSS Tax Saver Fund"

# facts: key -> (pill label or None, keyword pattern, canonical question, ELSS only)
# Order matters: the first matching pattern wins.
FACTS = {
    "expense_ratio": ("Expense ratio", r"\bexpense\s*ratios?\b|\bter\b|\bexpenses?\b", "What is the expense ratio of {f}?", False),
    "exit_load": ("Exit load", r"\bexit\s*loads?\b", "What is the exit load of {f}?", False),
    "min_sip": ("Minimum SIP amount", r"\bsips?\b", "What is the minimum SIP amount for {f}?", False),
    "riskometer": ("Riskometer", r"\brisk\s*-?\s*o?\s*meter\b|\brisk\s+(level|grade|category|profile)\b|\bhow risky\b",
                   "What is the riskometer level of {f}?", False),
    "benchmark": ("Benchmark", r"\bbench\s*marks?\b", "What is the benchmark of {f}?", False),
    "lock_in": ("Lock-in period", r"\block\s*-?\s*in\b", "What is the lock-in period of {f}?", True),
    "fund_manager": (None, r"\bfund\s*managers?\b|\bmanagers?\b|\bmanag(es|ed|ing)\b", "Who is the fund manager of {f}?", False),
    "aum": (None, r"\baum\b|\bfund\s*size\b|\bassets under management\b|\bcorpus\b", "What is the fund size (AUM) of {f}?", False),
    "min_lumpsum": (None, r"\blump\s*sum\b|\bminimum\s+(investment|amount)\b", "What is the minimum lumpsum investment for {f}?", False),
    "launch_date": (None, r"\blaunch(ed)?\b|\binception\b", "When was {f} launched?", False),
    "category": (None, r"\bcategory\b|\b(what|which) (type|kind) of (fund|scheme)\b", "What is the category of {f}?", False),
}
# fund-house (AMC) facts: no fund needed, the question is passed to answer() as typed
AMC_FACT = re.compile(r"\b(ceo|chief executive|managing director|md|cio|chief investment officer|sponsors?|"
                      r"trustees?|incorporat\w*|total aum|fund house)\b")

MSG = {
    "ask_fund": "Which HDFC fund would you like to check?",
    "ask_fact": "What would you like to know about {f}?",
}
# fixed replies, 2-3 variants each (picked at random), no LLM
REPLIES = {
    "greeting": [
        "Hi! I share quick facts about HDFC Mutual Fund schemes, like expense ratio, exit load and minimum SIP. "
        "Which fund house would you like to explore?",
        "Hello! I can help with quick facts about HDFC Mutual Fund schemes, facts only and no advice. "
        "Which fund house would you like to explore?",
        "Hey there! Ask me about HDFC Mutual Fund schemes, such as lock-in, riskometer or benchmark. "
        "Which fund house would you like to explore?",
    ],
    "closing": [
        "You're welcome, take care!",
        "Happy to help. Bye for now!",
        "Anytime! Come back whenever you need a quick fund fact.",
    ],
    "fallback": [
        "I can only share facts about one HDFC fund at a time, like expense ratio, exit load or minimum SIP.",
        "I'm built for quick facts about a single HDFC fund, such as expense ratio, exit load or lock-in period.",
        "I can help with one HDFC fund's facts at a time, like its riskometer, benchmark or minimum SIP.",
    ],
    "advice": [
        "I share facts only, not investment advice. I can tell you facts about HDFC mutual funds, like expense "
        "ratio, exit load or minimum SIP.",
        "I can't advise on what to invest in, I only share facts about HDFC mutual funds, such as expense ratio, "
        "exit load or minimum SIP.",
    ],
    "pii": [
        "Please don't share personal details here. I can't access accounts, but I'm happy to share HDFC "
        "mutual fund facts.",
        "For your safety, please don't share personal details here. I only share facts about HDFC mutual funds.",
    ],
}
LABEL = {"advice": "Advice question, facts only", "pii": "Personal details blocked"}

# --- message patterns (on lower-cased text without punctuation) ------------------------------
HOUSE = re.compile(r"^(hdfc|hdfc (mf|amc|mutual funds?))$")
CHANGE_FUND = re.compile(r"^(change|switch|choose|pick|select|another|different|other) (the )?(fund|scheme)s?$")
FUND_WORDS = re.compile(r"\b(hdfc|fund|funds|scheme|plan|direct|growth|the|select|selected|i|want|choose|pick)\b")
OTHER_AMC = re.compile(
    r"\b(sbi|icici|axis|kotak|nippon|aditya birla|absl|uti|mirae|parag parikh|ppfas|tata|dsp|franklin|"
    r"motilal|quant|canara robeco|edelweiss|bandhan|invesco|hsbc|sundaram|mahindra manulife|pgim|"
    r"bajaj finserv|lic mf|baroda bnp|navi|whiteoak|jm financial|union mutual|groww mutual)\b")
GREETING = re.compile(
    r"\b(hi+|hello+|hey+|hiya|namaste|greetings|good (morning|afternoon|evening|day)|"
    r"how (are|r) (you|u)( doing)?( today)?|hows it going|how is it going|whats up|wassup|sup|"
    r"who are (you|u)|what are (you|u)|what (can|do) (you|u) do|how (can|do|could) (you|u) help( me)?|"
    r"can (you|u) help( me)?|help( me)?|what is this|how does this work|start( over)?|menu)\b")
CLOSING = re.compile(
    r"\b(thanks?( (you|u))?( so much| a lot)?|thank u|thx|ty|bye+|goodbye|see (you|ya)|ok(ay)?|k|cool|great|"
    r"nice|got it|done|shukriya|dhanyavad)\b")
FILLER = {"there", "bot", "assistant", "please", "so", "and", "i", "im", "am", "fine", "good", "well", "doing",
          "today", "me", "you", "u", "very", "much", "again", "too", "also", "all", "for", "the", "help", "info"}


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower().replace("'", "").replace("’", ""))).strip()


@lru_cache(maxsize=1)
def alias_table() -> tuple[dict[str, str], float]:
    """{plain alias: canonical phrase} from config `aliases` (fund aliases map to the scheme name)."""
    cfg = get_config().get("aliases", {})
    table = {}
    for group in ("funds", "facts"):
        for canon, aliases in (cfg.get(group) or {}).items():
            for a in [canon, *aliases]:
                table[plain(a)] = plain(canon)
    return table, float(cfg.get("fuzzy_threshold", 0.85))


def normalize(p: str) -> str:
    """Replace fund / fact aliases and near-miss typos with their canonical phrase (routing only).
    Scans left to right, longest phrase (3 words) first; exact match, else difflib similarity for
    aliases and phrases of 5+ letters."""
    table, threshold = alias_table()
    fuzzy = [a for a in table if len(a) >= 5]
    words, out, i = p.split(), [], 0
    while i < len(words):
        for n in (3, 2, 1):
            phrase = " ".join(words[i:i + n])
            if len(words) - i < n:
                continue
            canon = table.get(phrase)
            if canon is None and len(phrase) >= 5:
                best = max(fuzzy, key=lambda a: SequenceMatcher(None, phrase, a).ratio(), default=None)
                if best and SequenceMatcher(None, phrase, best).ratio() >= threshold:
                    canon = table[best]
            if canon is not None:
                out.append(canon)
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


def display(fund: str | None) -> str | None:
    return FUNDS.get(fund) if fund else None


def only(pattern: re.Pattern, p: str) -> bool:
    """The whole message is made of these phrases (plus filler words)."""
    return bool(pattern.search(p)) and all(w in FILLER for w in pattern.sub(" ", p).split())


def fact_of(p: str) -> str | None:
    return next((key for key, (_, pat, _, _) in FACTS.items() if re.search(pat, p)), None)


def question_for(fund: str, fact: str) -> str:
    return FACTS[fact][2].format(f=display(fund))


def is_fund_selection(p: str) -> bool:
    """The message is just a fund name ("HDFC Mid Cap Fund", "mid cap"), not a question."""
    for pat in SCHEME_ALIASES:
        p = re.sub(pat, " ", p, flags=re.I)
    return not FUND_WORDS.sub(" ", p).split()


# --- pills (each carries its action, so a click is routed without any classification) ---------
def pill(label: str, action: str, fund: str | None = None, fact: str | None = None,
         message: str | None = None, kind: str = "option") -> dict:
    return {"label": label, "message": message or label, "kind": kind, "action": action, "fund": fund, "fact": fact}


def house_pills() -> list[dict]:
    return [pill("HDFC", "house")]


def fund_pills(fact: str | None = None) -> list[dict]:
    """Fund pills. With a pending fact ("what is the exit load?"), a click answers it directly."""
    if fact:
        return [pill(name, "fact", fund, fact, question_for(fund, fact)) for fund, name in FUNDS.items()]
    return [pill(name, "fund", fund) for fund, name in FUNDS.items()]


def question_pills(fund: str) -> list[dict]:
    out = [pill(label, "fact", fund, key, question_for(fund, key))
           for key, (label, _, _, elss) in FACTS.items() if label and (not elss or fund == ELSS)]
    return out + [pill("Change fund", "change_fund", kind="secondary")]


def default_pills(fund: str | None) -> list[dict]:
    return question_pills(fund) if fund else fund_pills()


# --- factual answers with a (fund, fact) cache ---------------------------------------------------
_CACHE: dict[tuple[str, str], tuple[float, dict]] = {}


def cached_answer(fund: str, fact: str) -> tuple[dict, bool]:
    """answer() for the canonical (fund, fact) question; factual answers are cached for cache.ttl_hours."""
    key, now = (fund, fact), time.time()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        return hit[1], True
    r = answer(question_for(fund, fact))
    if r["intent"] == "factual":
        ttl = get_config().get("cache", {}).get("ttl_hours", 24) * 3600
        _CACHE[key] = (now + ttl, {k: r[k] for k in ("intent", "text", "url", "last_updated")} | {"trace": {}})
    return r, False


# --- main entry --------------------------------------------------------------------------------
def reply(route: str, text: str, pills: list[dict], fund: str | None, trace: dict | None = None,
          **extra) -> dict:
    r = {"route": route, "intent": route, "text": text, "url": None, "last_updated": None,
         "label": LABEL.get(route), "pills": pills, "state": {"fund": fund}, "trace": trace or {}}
    r.update(extra)
    return r


def chat(message: str, state: dict | None = None, pill_click: dict | None = None) -> dict:
    fund = (state or {}).get("fund")
    fund = fund if fund in FUNDS else None
    q = re.sub(r"\s+", " ", message or "").strip()[: get_config()["guardrails"]["max_input_chars"]]
    trace = {"trace_id": uuid.uuid4().hex[:8], "at": datetime.now(IST).isoformat(timespec="seconds")}

    def routed(route: str, text: str, pills: list[dict], new_fund: str | None, **extra) -> dict:
        write_trace(trace | {"event": "chat_route", "route": route, "question": q,
                             "selected_fund": new_fund, "response": {"intent": route, "text": text}})
        return reply(route, text, pills, new_fund, trace | {"route": route}, **extra)

    def fixed(kind: str, pills: list[dict], **extra) -> dict:
        return routed(kind, random.choice(REPLIES[kind]), pills, fund, **extra)

    def factual(new_fund: str, fact: str, context_fund: str | None = None) -> dict:
        r, cached = cached_answer(new_fund, fact)
        if cached:
            write_trace(trace | {"event": "cache_hit", "fund": new_fund, "fact": fact})
        return reply("factual", r["text"], question_pills(new_fund), new_fund, r["trace"] or trace,
                     intent=r["intent"], url=r["url"], last_updated=r["last_updated"],
                     asked=question_for(new_fund, fact), cached=cached, context_fund=context_fund)

    def factual_as_typed() -> dict:                          # fund-house questions (no fund, no cache)
        r = answer(q)
        pills = fund_pills() if r["intent"] == "clarify" else default_pills(fund)
        return reply("factual", r["text"], pills, fund, r["trace"], intent=r["intent"], url=r["url"],
                     last_updated=r["last_updated"])

    # 0. empty -> answer flow (no LLM); PII -> fixed reply, the input is never logged or stored
    if not q:
        r = answer(q)
        return reply("factual", r["text"], default_pills(fund), fund, r["trace"], intent=r["intent"])
    pii_types = G.find_pii(q)
    if pii_types:
        write_trace({**trace, "event": "pii_rejected", "pii_types": pii_types})   # types only, never the text
        return reply("pii", random.choice(REPLIES["pii"]), default_pills(fund), fund, trace)

    # 1. pill click: its action is known, no classification
    if pill_click:
        action = pill_click.get("action")
        p_fund = pill_click.get("fund") if pill_click.get("fund") in FUNDS else None
        p_fact = pill_click.get("fact") if pill_click.get("fact") in FACTS else None
        if action in ("house", "change_fund"):
            return routed("fund_step", MSG["ask_fund"], fund_pills(), fund)
        if action == "fund" and p_fund:
            return routed("question_step", MSG["ask_fact"].format(f=display(p_fund)), question_pills(p_fund), p_fund)
        if action == "fact" and p_fund and p_fact:
            return factual(p_fund, p_fact)
        # unknown / invalid pill: fall through and treat it as typed text

    p = plain(q)
    n = normalize(p)                                         # aliases + typos -> canonical fund / fact names
    schemes = detect_schemes(n)

    # 2. typed navigation
    if CHANGE_FUND.match(p) or HOUSE.match(p):
        return routed("fund_step", MSG["ask_fund"], fund_pills(), fund)
    if len(schemes) == 1 and is_fund_selection(n):
        new = schemes[0]
        return routed("question_step", MSG["ask_fact"].format(f=display(new)), question_pills(new), new)

    # 3. other fund houses are never answered as an HDFC fund ("axis mid cap exit load")
    if OTHER_AMC.search(p):
        return fixed("fallback", default_pills(fund))

    # 4. advice -> fallback + one educational link
    if G.classify(q) == "advice":
        return fixed("advice", default_pills(fund), url=get_config()["links"]["education"], link_label="Learn more")

    # 5. facts
    fact = fact_of(n)
    if fact:
        if len(schemes) == 1:
            return factual(schemes[0], fact)                 # named fund becomes the selected fund
        if not schemes and fund:
            return factual(fund, fact, context_fund=display(fund))
        return routed("ask_fund", MSG["ask_fund"], fund_pills(fact), fund)   # no fund (or several)
    if AMC_FACT.search(p):
        return factual_as_typed()

    # 6. closing / greeting
    if only(CLOSING, p):
        return fixed("closing", [])
    if only(GREETING, p) or GREETING.match(p):
        return fixed("greeting", house_pills())

    # 7. anything else
    return fixed("fallback", default_pills(fund))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.chat "your message" [selected fund]')
    out = chat(sys.argv[1], {"fund": sys.argv[2] if len(sys.argv) > 2 else None})
    print(f"[{out['route']} / {out['intent']}] fund={out['state']['fund']}")
    print(out["text"])
    if out["url"]:
        print("Source:", out["url"])
    print("Pills:", [x["label"] for x in out["pills"]])
