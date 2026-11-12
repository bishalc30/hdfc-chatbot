"""Step 8 - UI (Streamlit).

Run from the project root:  streamlit run app/streamlit_app.py
"""
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from src.answerer import answer  # noqa: E402
from src.config import get_config, get_env, path  # noqa: E402

DISCLAIMER = (
    "Facts-only. No investment advice. Answers are drawn from official public sources and may not "
    "reflect the latest updates — please verify on the linked page. Mutual fund investments are "
    "subject to market risks; read all scheme-related documents carefully."
)
HIDDEN_PII = "‹message hidden: it contained personal details›"
INTENT_LABEL = {
    "factual": None, "advice": "Advice question — facts only",
    "performance": "Returns question — see official documents", "pii": "Personal details blocked",
    "not_available": "Not in sources", "clarify": "Needs scheme name", "error": "Error", "empty": None,
}

from PIL import Image  # noqa: E402

# Fully transparent 1x1 PNG, so no icon shows in the browser tab
st.set_page_config(page_title="HDFC MF Facts Assistant", page_icon=Image.open(ROOT / "app" / "blank.png"),
                   layout="centered", initial_sidebar_state="collapsed")
CFG = get_config()
SHOW_DEBUG = bool(CFG["ui"].get("show_debug", False))

GREEN, GREEN_DARK = "#1b5e20", "#144a18"
CSS = f"""
<style>
/* hide Streamlit chrome: header, menu, Deploy button, footer, sidebar */
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"] {{ display: none !important; }}

/* page */
.stApp {{ background: #f4f5f7; }}
.block-container, [data-testid="stMainBlockContainer"] {{
    max-width: 900px; padding-top: 3rem; padding-bottom: 3rem;
}}

/* header */
.app-title {{ font-size: 2rem; font-weight: 700; color: #111827; margin: 0 0 .35rem 0; line-height: 1.2; }}
.app-sub {{ color: #6b7280; font-size: 1rem; margin: 0 0 .9rem 0; }}
.pill {{
    display: inline-block; background: #e8f5e9; color: {GREEN}; border: 1px solid #c8e6c9;
    border-radius: 999px; padding: .2rem .75rem; font-size: .8rem; font-weight: 600; margin-bottom: 1.4rem;
}}
.section-label {{ color: #6b7280; font-size: .85rem; font-weight: 600; margin: .2rem 0 .4rem 0; }}

/* landing state: welcome line + example chips (wrap on narrow screens, never truncated) */
.welcome {{ text-align: center; color: #374151; font-size: 1.05rem; font-weight: 600; margin: .4rem 0 .3rem 0; }}
.st-key-examples {{ flex-wrap: wrap; justify-content: center; gap: .5rem; }}
.st-key-examples > div {{ max-width: 100%; }}
.st-key-examples button {{
    background: #ffffff; color: #1f2937; border: 1px solid #e5e7eb; border-radius: 10px;
    padding: .5rem .9rem; min-height: 0; max-width: 100%; box-shadow: 0 1px 2px rgba(0,0,0,.03);
}}
.st-key-examples button p {{ white-space: normal; overflow: visible; text-overflow: clip; text-align: center;
                             font-size: .92rem; line-height: 1.35; }}
.st-key-examples button:hover {{ border-color: {GREEN}; color: {GREEN}; background: #ffffff; }}
.note {{ color: #9ca3af; font-size: .8rem; margin: -.4rem 0 .4rem 0; }}

/* ask box: input + dark green Ask button */
.st-key-ask_box input {{ background: #ffffff; border-radius: 8px; }}
.st-key-ask_box [data-testid="stForm"] {{ border: none; padding: 0; }}
.st-key-ask_box [data-testid="stFormSubmitButton"] button {{
    background: {GREEN}; color: #ffffff; border: 1px solid {GREEN}; border-radius: 8px; font-weight: 600;
}}
.st-key-ask_box [data-testid="stFormSubmitButton"] button:hover {{
    background: {GREEN_DARK}; border-color: {GREEN_DARK}; color: #ffffff;
}}

/* answer cards */
.answer-card {{
    background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 1.1rem 1.25rem; margin: .9rem 0; box-shadow: 0 1px 3px rgba(0,0,0,.04);
}}
.answer-q {{ color: #6b7280; font-size: .9rem; margin-bottom: .5rem; }}
.answer-label {{ color: #92400e; background: #fef3c7; border-radius: 6px; font-size: .75rem;
                 padding: .1rem .5rem; display: inline-block; margin-bottom: .5rem; }}
.answer-text {{ color: #111827; font-size: 1rem; line-height: 1.55; }}
.answer-source {{ margin-top: .7rem; font-size: .9rem; color: #374151; word-break: break-all; }}
.answer-source a {{ color: {GREEN}; }}
.answer-updated {{ color: #9ca3af; font-size: .8rem; margin-top: .25rem; }}

/* clear button: plain text link style */
.st-key-clear button {{ background: transparent; border: none; color: #6b7280; padding: 0; min-height: 0; }}
.st-key-clear button:hover {{ color: {GREEN}; background: transparent; }}

/* footer */
.disclaimer {{ color: #9ca3af; font-size: .78rem; line-height: 1.5; }}
hr {{ margin: 1.5rem 0 1rem 0; border-color: #e5e7eb; }}
</style>
"""


# ---------------------------------------------------------------- helpers
def esc(text: str) -> str:
    """HTML-escape, keep line breaks, and stop '$' being read as LaTeX."""
    return html.escape(text or "").replace("$", "&#36;").replace("\n", "<br>")


def read_json(rel: str):
    p = path(rel)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_resource(show_spinner="Loading embedding model and vector DB…")
def warm_up():
    """Load the embedding model + Chroma collection once per server process."""
    from src.retriever import get_retriever
    return get_retriever()


def answer_card(question: str, r: dict) -> str:
    parts = ['<div class="answer-card">', f'<div class="answer-q">{esc(question)}</div>']
    label = INTENT_LABEL.get(r["intent"])
    if label:
        parts.append(f'<div class="answer-label">{esc(label)}</div>')
    parts.append(f'<div class="answer-text">{esc(r["text"])}</div>')
    if r.get("url"):
        url = html.escape(r["url"], quote=True)
        parts.append(f'<div class="answer-source"><b>Source:</b> '
                     f'<a href="{url}" target="_blank" rel="noopener">{url}</a></div>')
    if r.get("last_updated"):
        parts.append(f'<div class="answer-updated">Last updated from sources: {esc(r["last_updated"])}</div>')
    parts.append("</div>")
    return "".join(parts)


# ---------------------------------------------------------------- debug views (ui.show_debug)
def show_pipeline(r: dict) -> None:
    t = r["trace"]
    if r["intent"] == "pii":
        st.write("PII check blocked this message before retrieval. Only the detected types are logged.")
        st.json({"event": t.get("event"), "pii_types": t.get("pii_types"), "latency_ms": t.get("latency_ms")})
        return

    ret = t.get("retrieval", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Intent", t.get("intent", r["intent"]))
    c2.metric("Top similarity", f"{ret['hits'][0]['similarity']:.3f}" if ret.get("hits") else "—")
    c3.metric("Latency", f"{t.get('latency_ms', 0)} ms")

    tabs = st.tabs(["Retrieval", "Prompt", "LLM output", "Full trace"])
    with tabs[0]:
        if not ret:
            st.write("Retrieval was skipped (handled by the intent guardrail).")
        else:
            st.write(f"Schemes detected: `{ret.get('schemes_detected') or 'none'}` · "
                     f"Filter: `{ret.get('filter') or 'none'}` · "
                     f"Threshold: `{ret.get('threshold')}` · Confident: `{ret.get('confident')}`")
            st.dataframe(
                [{"similarity": h["similarity"], "scheme": h["scheme"] or "—", "section": h["section"],
                  "text": h["text"]} for h in ret.get("hits", [])],
                hide_index=True, width="stretch")
    with tabs[1]:
        msgs = t.get("llm", {}).get("prompt")
        if msgs:
            for m in msgs:
                st.markdown(f"**{m['role']}**")
                st.code(m["content"], language=None, wrap_lines=True)
        else:
            st.write("The LLM was not called for this question.")
    with tabs[2]:
        llm = t.get("llm", {})
        if llm:
            st.write(f"Provider: `{llm.get('provider')}` · Model: `{llm.get('model')}` · "
                     f"Validation: `{t.get('validation', '—')}`")
            if llm.get("error"):
                st.error(llm["error"])
            st.code(llm.get("raw_output", ""), language=None, wrap_lines=True)
        else:
            st.write("The LLM was not called for this question.")
    with tabs[3]:
        st.json(t, expanded=False)


def pipeline_status() -> None:
    fetch = read_json("data/01_raw/fetch_report.json")
    load = read_json("data/02_cleaned/load_report.json")
    chunks = read_json("data/03_chunks/stats.json")
    emb = read_json("data/04_embeddings/manifest.json")
    index = read_json("data/05_vectordb/index_report.json")

    rows = [
        ("1 · Scrape", f"{sum(s['result'] in ('fetched', 'unchanged') for s in fetch['sources'])}"
                       f"/{len(fetch['sources'])} pages" if fetch else "not run"),
        ("2 · Load", f"{sum(d.get('ok', False) for d in load)}/{len(load)} docs OK" if load else "not run"),
        ("3 · Chunk", f"{chunks['total_chunks']} chunks" if chunks else "not run"),
        ("4 · Embed", f"{emb['count']} × {emb['dim']}d" if emb else "not run"),
        ("5 · Index", f"{index['count']} items" if index else "not run"),
    ]
    for name, value in rows:
        st.markdown(f"**{name}** — {value}")
    env = get_env()
    st.caption(f"LLM: **{env['provider']}** / `{env['model']}` · Embeddings: `{CFG['embedding']['model_name']}`")
    for label, data in [("Chunks", chunks), ("Embeddings", emb), ("Vector DB", index)]:
        if data:
            st.caption(label)
            st.json(data, expanded=False)


# ---------------------------------------------------------------- page
st.markdown(CSS, unsafe_allow_html=True)

st.markdown('<div class="app-title">HDFC Mutual Fund Facts Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Ask factual questions about HDFC Mutual Fund schemes — expense ratio, '
            'exit load, minimum SIP, lock-in, riskometer, benchmark, fund manager and statements.</div>',
            unsafe_allow_html=True)
st.markdown('<span class="pill">Facts-only. No investment advice.</span>', unsafe_allow_html=True)

try:
    env = get_env()
    if not env["api_key"]:
        st.warning(f"{env['provider'].upper()}_API_KEY is not set in .env — factual answers will fail.")
except ValueError as e:
    st.error(str(e))

try:
    warm_up()
except Exception as e:  # missing index, model mismatch, ...
    st.error(f"The knowledge base isn't ready: {e}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# landing state: welcome line + example chips, only until the first answer
clicked = None
landing = st.empty()
if not st.session_state.history:
    with landing.container():
        st.markdown(f'<div class="welcome">{esc(CFG["ui"].get("welcome", ""))}</div>', unsafe_allow_html=True)
        with st.container(key="examples", horizontal=True, horizontal_alignment="center", gap="small"):
            for i, q in enumerate(CFG["ui"]["example_questions"]):
                if st.button(q, key=f"example_{i}", width="content"):
                    clicked = q

# question input + Ask button
typed = None
with st.container(key="ask_box"):
    with st.form("ask_form", clear_on_submit=True, border=False):
        col_in, col_btn = st.columns([6, 1], vertical_alignment="bottom")
        text = col_in.text_input("Your question", placeholder="Type your question…",
                                 label_visibility="collapsed")
        if col_btn.form_submit_button("Ask", width="stretch"):
            typed = text
st.markdown('<div class="note">Facts-only. No investment advice.</div>', unsafe_allow_html=True)

question = typed or clicked
if question is not None:
    landing.empty()   # hide welcome + examples as soon as a question is asked
    with st.spinner("Searching official sources…"):
        try:
            resp = answer(question)
        except Exception as e:
            resp = {"intent": "error", "text": f"Something went wrong: {e}", "url": None,
                    "last_updated": None, "trace": {}}
    shown = HIDDEN_PII if resp["intent"] == "pii" else question
    st.session_state.history.append({"question": shown, "response": resp})

# answers: newest first, each in a white card
for turn in reversed(st.session_state.history):
    st.markdown(answer_card(turn["question"], turn["response"]), unsafe_allow_html=True)

if st.session_state.history:
    with st.container(key="clear"):
        if st.button("Clear answers"):
            st.session_state.history = []
            st.rerun()

if SHOW_DEBUG:
    if st.session_state.history and st.session_state.history[-1]["response"].get("trace"):
        with st.expander("🔍 Show pipeline (last answer)"):
            show_pipeline(st.session_state.history[-1]["response"])
    with st.expander("RAG pipeline status"):
        pipeline_status()

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="disclaimer">{esc(DISCLAIMER)}</div>', unsafe_allow_html=True)
