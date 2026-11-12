"""Vercel serverless API (FastAPI) for the HDFC MF Facts Assistant.

  GET  /api/config  -> welcome line, example questions, disclaimer, badge text
  POST /api/ask     -> {"question": "..."} -> answer card data
  POST /api/chat    -> {"message": "...", "state": {"fund": ...}} -> reply + pills + new state
  GET  /api/health  -> index/model status

Uses the same answer flow as the Streamlit app (src/answerer.py), with the "serving"
retrieval backend (ONNX model + exported index) so no PyTorch/chromadb is needed.

Local run:  uvicorn api.index:app --reload   (then open http://127.0.0.1:8000)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Serverless defaults (can be overridden by environment variables)
os.environ.setdefault("RETRIEVAL_BACKEND", "serving")
if os.getenv("VERCEL"):
    os.environ.setdefault("LOG_DIR", "/tmp/mf-faq-logs")   # only /tmp is writable on Vercel

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from src.answerer import answer  # noqa: E402
from src.chat import chat  # noqa: E402
from src.config import get_config  # noqa: E402

DISCLAIMER = (
    "Facts-only. No investment advice. Answers are drawn from official public sources and may not "
    "reflect the latest updates — please verify on the linked page. Mutual fund investments are "
    "subject to market risks; read all scheme-related documents carefully."
)
HIDDEN_PII = "‹message hidden: it contained personal details›"
INTENT_LABEL = {
    "advice": "Advice question — facts only",
    "performance": "Returns question — see official documents",
    "pii": "Personal details blocked",
    "not_available": "Not in sources",
    "clarify": "Needs scheme name",
    "error": "Error",
}

app = FastAPI(title="HDFC MF Facts Assistant API", docs_url=None, redoc_url=None)
allowed = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if allowed:
    app.add_middleware(CORSMiddleware, allow_origins=allowed, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"])


class AskRequest(BaseModel):
    question: str = Field(default="", max_length=2000)


class ChatState(BaseModel):
    fund: str | None = Field(default=None, max_length=100)


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=2000)
    state: ChatState = Field(default_factory=ChatState)


@app.get("/api/config")
def config():
    cfg = get_config()
    return {"welcome": cfg["ui"].get("welcome"), "example_questions": cfg["ui"]["example_questions"],
            "disclaimer": DISCLAIMER,
            "badge": "Facts-only. No investment advice."}


@app.get("/api/health")
def health():
    try:
        from src.retriever import get_retriever
        r = get_retriever()
        return {"ok": True, "backend": os.getenv("RETRIEVAL_BACKEND"), "embedding_model": r.model_name}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=503)


@app.post("/api/ask")
def ask(req: AskRequest):
    try:
        r = answer(req.question)
    except Exception as e:
        return JSONResponse({"intent": "error", "text": f"Something went wrong: {e}", "url": None,
                             "last_updated": None, "label": INTENT_LABEL["error"],
                             "question": req.question}, status_code=500)
    body = {
        "intent": r["intent"],
        "text": r["text"],
        "url": r["url"],
        "last_updated": r["last_updated"],
        "label": INTENT_LABEL.get(r["intent"]),
        "question": HIDDEN_PII if r["intent"] == "pii" else req.question,   # never echo PII back
    }
    if get_config()["ui"].get("show_debug"):
        body["trace"] = r["trace"]
    return body


@app.post("/api/chat")
def chat_turn(req: ChatRequest):
    """Guided conversation: routes the message (small talk / navigation / comparison / advice /
    factual), returns the reply, the pills to show and the new state for the client to send back."""
    try:
        r = chat(req.message, req.state.model_dump())
    except Exception as e:
        return JSONResponse({"route": "error", "intent": "error", "text": f"Something went wrong: {e}",
                             "url": None, "last_updated": None, "label": INTENT_LABEL["error"],
                             "pills": [], "state": req.state.model_dump(), "question": req.message},
                            status_code=500)
    body = {k: r.get(k) for k in ("route", "intent", "text", "url", "last_updated", "pills", "state",
                                  "context_fund")}
    body["label"] = r.get("label") or INTENT_LABEL.get(r["intent"])
    body["question"] = HIDDEN_PII if r["intent"] == "pii" else req.message   # never echo PII back
    if get_config()["ui"].get("show_debug"):
        body["trace"] = r["trace"]
    return body


# Serve the page (public/index.html) from the app itself, on Vercel and locally.
# Registered after the /api routes, so those take precedence.
if (ROOT / "public").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=ROOT / "public", html=True), name="static")
