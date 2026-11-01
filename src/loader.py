"""Step 3 - Load documents (parse, clean, validate).

Reads data/01_raw/<slug>.html + <slug>.meta.json and writes:
  data/02_cleaned/<slug>.json       one clean document per source
  data/02_cleaned/load_report.json  fields found / missing, sections, ok flag

Groww pages embed their data in <script id="__NEXT_DATA__"> (props.pageProps):
  scheme pages -> mfServerSideData
  AMC page     -> amcMainPageData[1] (contentList, key_information)
Other (official) pages are parsed from visible headings and paragraphs.

Returns/performance data (stats, return_stats, sip_return, simple_return, analysis,
peerComparison, NAV, holdings) is never copied into the cleaned documents.
"""
import glob
import html as htmllib
import json
import re

from bs4 import BeautifulSoup

from src.config import path

# Anything performance-related is dropped from fields and from section text.
PERF_KEY = re.compile(
    r"return|cagr|xirr|(?<![a-z])nav(?![a-z])|(?<![a-z])stats?(?![a-z])|rank|analysis|peer|"
    r"holding|sharpe|(?<![a-z])(alpha|beta)(?![a-z])", re.I)
PERF_SENTENCE = re.compile(
    r"\breturns?\b(?!\s+index)[^.]*\d|\bcagr\b|\bxirr\b|\bannuali[sz]ed\b|"
    r"\d+(\.\d+)?\s*%\s*(returns?|p\.?\s?a\.?|per annum)|\bnav\b[^.]*\d", re.I)

REQUIRED_SCHEME_FIELDS = ["expense_ratio", "exit_load", "min_sip", "riskometer", "benchmark"]


# ---------------------------------------------------------------- helpers
def clean_text(text: str) -> str:
    """Unescape HTML entities, collapse whitespace, drop duplicate lines."""
    text = htmllib.unescape(text or "").replace("\xa0", " ")
    seen, lines = set(), []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line and line.lower() not in seen:
            seen.add(line.lower())
            lines.append(line)
    return "\n".join(lines)


def drop_performance_sentences(text: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(s for s in sentences if not PERF_SENTENCE.search(s)).strip()


def html_to_sections(fragment: str, default_heading: str) -> list[dict]:
    """Group visible text under its nearest h1-h3 heading."""
    soup = BeautifulSoup(fragment, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form", "button"]):
        tag.decompose()
    sections, heading, buf = [], default_heading, []

    def flush():
        text = drop_performance_sentences(clean_text("\n".join(buf)))
        if text and not PERF_KEY.search(heading):
            sections.append({"heading": heading, "text": text})

    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        txt = el.get_text(" ", strip=True)
        if not txt:
            continue
        if el.name in ("h1", "h2", "h3"):
            flush()
            heading, buf = clean_text(txt), []
        else:
            buf.append(txt)
    flush()
    return sections


def next_data(raw_html: str) -> dict | None:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', raw_html, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def rupees(value) -> str | None:
    if value in (None, ""):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"₹{num:,.0f}" if num.is_integer() else f"₹{num:,.2f}"


# ---------------------------------------------------------------- Groww scheme page
def format_lock_in(lock_in: dict | None, extra: dict | None) -> str:
    lock_in = lock_in or {}
    parts = []
    for unit in ("years", "months", "days"):
        n = lock_in.get(unit)
        if n:
            parts.append(f"{n} {unit if n != 1 else unit[:-1]}")
    if not parts and (extra or {}).get("lock_in_yrs"):
        parts.append(f"{extra['lock_in_yrs']} years")
    return " ".join(parts) if parts else "No lock-in period"


def riskometer(s: dict) -> str | None:
    # The visible riskometer matches return_stats[0].risk ("Very High").
    # nfo_risk is a different (NFO-time) value and is NOT used.
    for row in s.get("return_stats") or []:
        if isinstance(row, dict) and row.get("risk"):
            return row["risk"]
    m = re.search(r"Risk is ([A-Za-z ]+?)\s*$", s.get("meta_desc") or "")
    return m.group(1).strip() if m else None


def parse_groww_scheme(s: dict) -> tuple[dict, list[dict], dict]:
    managers = []
    for m in s.get("fund_manager_details") or []:
        name = (m or {}).get("person_name")
        if name and name not in managers:
            managers.append(name)
    if not managers and s.get("fund_manager"):
        managers = [s["fund_manager"]]

    exit_load = clean_text(s.get("exit_load") or "")
    if exit_load.lower() in ("nil", "0", "none"):
        exit_load = "Nil (no exit load)"

    fields = {
        "scheme_full_name": s.get("scheme_name"),
        "fund_house": s.get("fund_house"),
        "category": " – ".join(x for x in (s.get("category"), s.get("sub_category")) if x) or None,
        "plan": " ".join(x for x in (s.get("plan_type") and f"{s['plan_type']} Plan", s.get("scheme_type")) if x) or None,
        "expense_ratio": f"{s['expense_ratio']}%" if s.get("expense_ratio") not in (None, "") else None,
        "exit_load": exit_load or None,
        "min_sip": rupees(s.get("min_sip_investment")),
        "min_lumpsum": rupees(s.get("min_investment_amount")),
        "min_additional_investment": rupees(s.get("mini_additional_investment")),
        "lock_in": format_lock_in(s.get("lock_in"), s.get("additional_details")),
        "riskometer": riskometer(s),
        "benchmark": s.get("benchmark_name") or s.get("benchmark"),
        "fund_size_aum": f"{rupees(s['aum'])} Cr" if s.get("aum") not in (None, "") else None,
        "fund_manager": ", ".join(managers) or None,
        "launch_date": s.get("launch_date"),
        "registrar": (s.get("rta_details") or {}).get("rta_name") or s.get("registrar_agent"),
        "stamp_duty": s.get("stamp_duty"),
    }
    fields = {k: str(v) for k, v in fields.items() if v not in (None, "") and not PERF_KEY.search(k)}

    sections = []
    about = drop_performance_sentences(clean_text(s.get("description") or ""))
    if about:
        sections.append({"heading": f"About {s.get('scheme_name', 'the scheme')}", "text": about})

    links = {k: v for k, v in {"sid": s.get("sid_url"), "amc_page": s.get("amc_page_url")}.items() if v}
    return fields, sections, links


# ---------------------------------------------------------------- Groww AMC page
def parse_groww_amc(pp: dict) -> tuple[dict, list[dict], dict]:
    blocks = pp.get("amcMainPageData") or []
    info = next((b for b in blocks if isinstance(b, dict) and "contentList" in b), {})

    sections = []
    for item in info.get("contentList") or []:
        for sec in html_to_sections(item.get("content", ""), clean_text(item.get("title", "Overview"))):
            sections.append(sec)

    k = info.get("key_information") or {}
    as_list = lambda v: ", ".join(v) if isinstance(v, list) else v
    fields = {
        "fund_house": k.get("fund_house"),
        "incorporation_date": (k.get("amc_incorporation_date") or "")[:10] or None,
        "sponsor": as_list(k.get("sponsor_name")),
        "trustee_organisation": k.get("trustee_organisation"),
        "ceo": k.get("ceo"),
        "managing_director": k.get("md"),
        "chief_investment_officer": k.get("cio"),
        "total_aum": f"{rupees(k['total_aum'])} Cr" if k.get("total_aum") else None,
    }
    fields = {k: str(v) for k, v in fields.items() if v not in (None, "")}
    return fields, sections, {}


# ---------------------------------------------------------------- fallback for scheme pages
LABELS = {
    "expense ratio": "expense_ratio", "exit load": "exit_load", "min. sip amount": "min_sip",
    "minimum sip": "min_sip", "min. for 1st investment": "min_lumpsum", "lock-in": "lock_in",
    "benchmark": "benchmark", "fund size": "fund_size_aum",
}


def parse_by_labels(raw_html: str) -> dict:
    """Used only if __NEXT_DATA__ is missing: read 'Label  value' pairs from visible text."""
    soup = BeautifulSoup(raw_html, "lxml")
    fields = {}
    for node in soup.find_all(string=True):
        key = node.strip().lower().rstrip(":")
        if key in LABELS and LABELS[key] not in fields and node.parent:
            value = node.parent.find_next(string=lambda t: t and t.strip() and t.strip().lower() != key)
            if value:
                fields[LABELS[key]] = clean_text(value)
    m = re.search(r"rated ([A-Za-z ]+?) risk", soup.get_text(" "), re.I)
    if m:
        fields["riskometer"] = m.group(1).strip()
    return fields


# ---------------------------------------------------------------- per-document
def load_document(meta: dict, raw_html: str) -> dict:
    data = next_data(raw_html)
    pp = (data or {}).get("props", {}).get("pageProps", {})
    fields, sections, links, parser = {}, [], {}, "html"

    if meta["source_type"] == "scheme" and "mfServerSideData" in pp:
        fields, sections, links = parse_groww_scheme(pp["mfServerSideData"])
        parser = "groww_next_data"
    elif meta["source_type"] == "amc_overview" and "amcMainPageData" in pp:
        fields, sections, links = parse_groww_amc(pp)
        parser = "groww_next_data"
    else:
        if meta["source_type"] == "scheme":
            fields, parser = parse_by_labels(raw_html), "html_labels"
        title = BeautifulSoup(raw_html, "lxml").title
        sections = html_to_sections(raw_html, clean_text(title.get_text()) if title else meta["slug"])

    return {
        "slug": meta["slug"],
        "url": meta["url"],
        "source_type": meta["source_type"],
        "scheme_name": meta.get("scheme_name", ""),
        "fetched_at": meta["fetched_at"],
        "parser": parser,
        "fields": fields,
        "sections": sections,
        "official_links": links,
    }


def validate(doc: dict) -> dict:
    problems = []
    if not doc.get("url") or not doc.get("fetched_at"):
        problems.append("missing url/fetched_at")
    missing = []
    if doc["source_type"] == "scheme":
        need = REQUIRED_SCHEME_FIELDS + (["lock_in"] if "elss" in doc["scheme_name"].lower() else [])
        missing = [f for f in need if f not in doc["fields"]]
        if "elss" in doc["scheme_name"].lower() and doc["fields"].get("lock_in") == "No lock-in period":
            problems.append("ELSS scheme without a lock-in")
    if not doc["fields"] and not doc["sections"]:
        problems.append("empty document")
    # Check each field/section on its own: joining them creates false matches across fields
    # (e.g. "...Total Return Index" followed by the AUM figure).
    leaked = [k for k, v in doc["fields"].items() if PERF_SENTENCE.search(v)]
    leaked += [s["heading"] for s in doc["sections"] if PERF_SENTENCE.search(s["text"])]
    if leaked:
        problems.append(f"performance text leaked in: {leaked}")
    return {
        "slug": doc["slug"],
        "parser": doc["parser"],
        "fields_found": sorted(doc["fields"]),
        "missing_fields": missing,
        "sections": len(doc["sections"]),
        "problems": problems,
        "ok": not missing and not problems,
    }


def run() -> list[dict]:
    out_dir = path("data/02_cleaned")
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_files = sorted(glob.glob(str(path("data/01_raw/*.meta.json"))))
    if not meta_files:
        raise SystemExit("No raw pages found in data/01_raw - run `python -m src.pipeline scrape` first.")

    report = []
    for meta_path in meta_files:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        html_path = meta_path.replace(".meta.json", ".html")
        try:
            with open(html_path, encoding="utf-8") as f:
                raw_html = f.read()
        except FileNotFoundError:
            report.append({"slug": meta["slug"], "ok": False, "problems": ["raw html missing"]})
            print(f"  MISSING  {meta['slug']}.html")
            continue

        doc = load_document(meta, raw_html)
        result = validate(doc)
        report.append(result)
        with open(out_dir / f"{doc['slug']}.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

        status = "OK  " if result["ok"] else "WARN"
        detail = "" if result["ok"] else f"  missing={result['missing_fields']} problems={result['problems']}"
        print(f"  {status}  {doc['slug']:<16} fields={len(doc['fields']):>2} sections={len(doc['sections'])}{detail}")

    with open(out_dir / "load_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Done: {sum(r['ok'] for r in report)}/{len(report)} documents OK")
    return report


if __name__ == "__main__":
    run()
