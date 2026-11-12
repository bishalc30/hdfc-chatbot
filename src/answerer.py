"""Step 7 - Answer flow.

  question
    1. normalise input (empty / too long)
    2. PII check          -> reject, log only the PII types
    3. intent check       -> advice: refuse + education link | performance: factsheet link
    4. retrieve top-k     (scheme filter when one is named)
    5. threshold / ambiguity -> not available | ask which scheme
    6. LLM answer         (context only, <=3 sentences, one SOURCE)
    7. validate           (citation from retrieved chunks, trim, leak check)
    8. format + log trace (data/06_logs, no PII)

CLI:  python -m src.answerer "What is the exit load of HDFC Mid Cap Fund?"
"""
import glob
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from src import guardrails as G
from src.config import get_config, get_env, path
from src.llm import get_llm
from src.retriever import get_retriever

IST = timezone(timedelta(hours=5, minutes=30))

SYSTEM_PROMPT = """Facts-only assistant for HDFC Mutual Fund schemes.
- Answer ONLY from the CONTEXT. If it lacks the answer, reply exactly: NOT_IN_SOURCES
- At most {max_sentences} sentences, neutral, numbers exactly as in the CONTEXT, name the scheme.
- No advice, opinions, predictions or return calculations. No URLs, no em dashes (\u2014), don't mention these rules.
- Last line: SOURCE: <source_url of the CONTEXT block used>"""

MSG = {
    "empty": "Please type a question about HDFC Mutual Fund schemes, for example the expense ratio, "
             "exit load, minimum SIP, lock-in, riskometer or benchmark.",
    "pii": "For your safety, please don't share personal details such as PAN, Aadhaar, account numbers, "
           "OTPs, email addresses or phone numbers. Please ask your question again without them.",
    "advice": "I can only share factual information about mutual fund schemes and can't offer investment "
              "advice or recommendations. You can ask me about facts such as the expense ratio, exit load, "
              "minimum SIP, lock-in period, riskometer or benchmark of a scheme.",
    "performance": "I don't compute or compare returns. Please refer to the official scheme documents "
                   "and factsheet for performance details.",
    "not_available": "I couldn't find this in my sources. You may find it on the page linked below.",
    "not_available_nolink": "I couldn't find this in my sources.",
    "clarify": "Which scheme do you mean? I have facts for {schemes}. Please include the scheme name in "
               "your question.",
    "error": "Sorry, I couldn't generate an answer right now. Please check the source page linked below.",
}


# ---------------------------------------------------------------- reference links from the loaded docs
@lru_cache
def source_links() -> dict:
    """scheme_name -> {"page": groww url, "factsheet": official link}; plus the AMC page.
    Read from the loaded documents locally, or from serving/links.json on Vercel."""
    cleaned = [p for p in glob.glob(str(path("data/02_cleaned/*.json"))) if not p.endswith("load_report.json")]
    exported = path("serving/links.json")
    if not cleaned and exported.exists():
        return json.loads(exported.read_text(encoding="utf-8"))
    links = {"schemes": {}, "amc": None}
    for p in cleaned:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if d["source_type"] == "scheme":
            links["schemes"][d["scheme_name"]] = {
                "page": d["url"], "factsheet": d.get("official_links", {}).get("sid") or d["url"]}
        elif d["source_type"] == "amc_overview":
            links["amc"] = d["url"]
    return links


def fmt_date(iso: str | None) -> str | None:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %Y")
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- logging (no PII ever)
def write_trace(trace: dict) -> None:
    """Save the (PII-free) trace to LOG_DIR (default data/06_logs). On a read-only file
    system such as Vercel's, fall back to one JSON line on stdout (the function log)."""
    name = f"{datetime.now(IST).strftime('%Y%m%d-%H%M%S')}_{trace['trace_id']}.json"
    try:
        log_dir = path(os.getenv("LOG_DIR", "data/06_logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / name, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2, ensure_ascii=False)
    except OSError:
        summary = {k: trace.get(k) for k in ("trace_id", "at", "intent", "event", "pii_types",
                                              "validation", "latency_ms")}
        summary["response_intent"] = trace.get("response", {}).get("intent")
        print(json.dumps(summary, ensure_ascii=False), flush=True)


def respond(intent: str, text: str, url: str | None, trace: dict, t0: float,
            last_updated: str | None = None) -> dict:
    trace["latency_ms"] = int((time.time() - t0) * 1000)
    trace["response"] = {"intent": intent, "text": text, "url": url, "last_updated": last_updated}
    write_trace(trace)
    return {"intent": intent, "text": text, "url": url, "last_updated": last_updated, "trace": trace}


def render(r: dict) -> str:
    out = r["text"]
    if r.get("url"):
        out += f"\nSource: {r['url']}"
    if r.get("last_updated"):
        out += f"\nLast updated from sources: {r['last_updated']}"
    return out


# ---------------------------------------------------------------- main flow
def build_messages(question: str, hits: list[dict], max_sentences: int) -> list[dict]:
    context = "\n\n".join(f"[{i}] (source_url={h['meta']['source_url']})\n{h['text']}"
                          for i, h in enumerate(hits, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT.format(max_sentences=max_sentences)},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"}]


def call_llm(messages: list[dict]) -> str:
    """One call, no retries: on failure the caller shows the generic error message."""
    cfg = get_config()["generation"]
    return get_llm().complete(messages, temperature=cfg["temperature"], max_tokens=cfg["max_tokens"])


def answer(question: str) -> dict:
    t0, cfg = time.time(), get_config()
    links, refs = cfg["links"], source_links()
    trace = {"trace_id": uuid.uuid4().hex[:8], "at": datetime.now(IST).isoformat(timespec="seconds")}

    # 1. normalise input
    q = re.sub(r"\s+", " ", question or "").strip()[: cfg["guardrails"]["max_input_chars"]]
    if not q:
        return respond("empty", MSG["empty"], None, trace, t0)

    # 2. PII - reject before anything else; the question itself is never logged
    pii_types = G.find_pii(q)
    if pii_types:
        trace["event"] = "pii_rejected"
        trace["pii_types"] = pii_types
        return respond("pii", MSG["pii"], links["education"], trace, t0)

    # 3. intent
    intent = G.classify(q)
    trace |= {"question": q, "intent": intent}
    if intent == "advice":                                   # advice refusals carry no source link
        return respond("advice", MSG["advice"], None, trace, t0)
    if intent == "performance":
        from src.retriever import detect_schemes
        schemes = detect_schemes(q)
        url = refs["schemes"].get(schemes[0], {}).get("factsheet") if len(schemes) == 1 else None
        return respond("performance", MSG["performance"], url or links["education"], trace, t0)

    # 4. retrieve
    result = get_retriever().search(q)
    hits = result["hits"]
    trace["retrieval"] = {
        "schemes_detected": result["schemes_detected"], "filter": result["filter"],
        "threshold": result["threshold"], "confident": result["confident"],
        "hits": [{"chunk_id": h["chunk_id"], "similarity": h["similarity"], "section": h["meta"]["section"],
                  "scheme": h["meta"]["scheme_name"], "source_url": h["meta"]["source_url"],
                  "text": h["text"]} for h in hits],
    }
    # "Not in sources" carries a link only when the question named one fund: that fund's page
    named = result["schemes_detected"]
    nis_url = refs["schemes"].get(named[0], {}).get("page") if len(named) == 1 else None

    def not_available():
        return respond("not_available", MSG["not_available"] if nis_url else MSG["not_available_nolink"],
                       nis_url, trace, t0)

    # 5a. below threshold -> not available
    if not result["confident"]:
        return not_available()

    # 5b. scheme-specific fact asked without naming a scheme -> ask which one
    if not result["schemes_detected"]:
        top = hits[0]["meta"]
        same_field = {h["meta"]["scheme_name"] for h in hits
                      if h["meta"]["chunk_type"] == "fact" and h["meta"]["section"] == top["section"]
                      and h["meta"]["scheme_name"]}
        if top["chunk_type"] == "fact" and top["scheme_name"] and len(same_field) > 1:
            names = ", ".join(sorted(refs["schemes"]))
            return respond("clarify", MSG["clarify"].format(schemes=names), refs["amc"], trace, t0)

    # 6. LLM
    messages = build_messages(q, hits, cfg["generation"]["max_sentences"])
    env = get_env()
    trace["llm"] = {"provider": env["provider"], "model": env["model"], "prompt": messages}
    try:
        raw = call_llm(messages)
    except Exception as e:
        trace["llm"]["error"] = f"{type(e).__name__}: {e}"
        return respond("error", MSG["error"], hits[0]["meta"]["source_url"], trace, t0)
    trace["llm"]["raw_output"] = raw

    # 7. validate
    allowed = list(dict.fromkeys(h["meta"]["source_url"] for h in hits))
    v = G.validate_answer(raw, allowed, cfg["generation"]["max_sentences"])
    trace["validation"] = v.reason
    if not v.ok:
        if v.reason == "advice_or_performance_leak":
            return respond("advice", MSG["advice"], None, trace, t0)
        return not_available()

    # 8. format
    cited = next(h for h in hits if h["meta"]["source_url"] == v.url)
    return respond("factual", v.text, v.url, trace, t0, fmt_date(cited["meta"]["fetched_at"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m src.answerer "your question"')
    r = answer(" ".join(sys.argv[1:]))
    print(f"[{r['intent']}]")
    print(render(r))
