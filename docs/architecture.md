# Architecture: Mutual Fund FAQ Assistant (RAG Demo)

| Field | Value |
|---|---|
| Based on | `docs/PRD.md` (v1.0) |
| Status | Draft v1.0 |
| Date | 2026-11-01 |

---

## 1. Purpose

This document turns the PRD into a technical design: components, module layout, data contracts, request flow, configuration, and guardrail placement. The system has two separate pipelines:

1. **Ingestion pipeline (offline, run by phase):** URLs → raw pages → cleaned docs → chunks → embeddings → ChromaDB. Every phase writes its output to disk.
2. **Query pipeline (online, per request):** user question → guardrails → retrieval → LLM → post-checks → answer with one citation.

The two pipelines share only these things: `config.yaml`, the embedding model, and the persisted ChromaDB collection.

## 2. High-Level Diagram

```
                         ┌──────────────────────── INGESTION (offline) ────────────────────────┐
                         │                                                                      │
 sources.csv ──► [P1 Loader] ──► data/01_raw ──► [P1 Cleaner] ──► data/02_cleaned               │
                         │                                              │                       │
                         │                                              ▼                       │
                         │                                     [P2 Chunker] ──► data/03_chunks  │
                         │                                              │                       │
                         │                                              ▼                       │
                         │                                     [P3 Embedder] ─► data/04_embeddings
                         │                                              │                       │
                         │                                              ▼                       │
                         │                                     [P4 Indexer] ──► data/05_vectordb (ChromaDB)
                         └──────────────────────────────────────────────┼───────────────────────┘
                                                                        │ (shared, read-only)
                         ┌──────────────────────── QUERY (online) ──────┼───────────────────────┐
                         │                                              │                       │
 User ─► [Streamlit UI] ─► [PII Guard] ─► [Intent Router] ─► [Retriever] ─► [Prompt Builder]    │
            ▲            │      │               │                 │              │              │
            │            │   reject         refuse/redirect   below threshold    ▼              │
            │            │      │               │             → "not in sources" [LLM Client]   │
            │            │      ▼               ▼                 │      (Groq | Gemini | OpenAI│
            │            │  [Response Formatter] ◄────────────────┴──── | Anthropic | Mistral)  │
            │            │      │                                        │                      │
            │            │      │  ◄──────── [Answer Validator] ◄────────┘                      │
            └────────────┼──────┘                │                                              │
                         │                       ▼                                              │
                         │               data/06_retrieval_logs (PII-free)                      │
                         └──────────────────────────────────────────────────────────────────────┘
```

## 3. Technology Stack

| Concern | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Scraping | `httpx` + `beautifulsoup4`; `playwright` as fallback | Parse embedded `__NEXT_DATA__` JSON first |
| Chunking | Custom structure-aware chunker + token counting with the HF tokenizer | The tokenizer must match the embedding model |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` (384-dim) | Normalized vectors, cosine similarity |
| Vector DB | `chromadb` (PersistentClient) | `hnsw:space = cosine` |
| LLM | Provider adapters: Groq (default), Gemini, OpenAI, Anthropic, Mistral | Selected through `.env` |
| Config | `pyyaml` + `pydantic-settings` + `python-dotenv` | Typed, checked at startup |
| UI | Streamlit | Chat plus a "Show pipeline" panel |
| CLI | `typer` | Runs each ingestion phase separately |
| Tests | `pytest` | Guardrail, chunking, and end-to-end Q&A tests |

## 4. Repository Layout

```
RAG Chatbot/
├── config.yaml                 # non-secret, changeable settings
├── .env.example                # LLM_PROVIDER, LLM_MODEL, *_API_KEY
├── sources.csv                 # 15–25 verified URLs (deliverable D2)
├── README.md                   # deliverable D3
├── sample_qa.md                # deliverable D4
├── requirements.txt
├── docs/
│   ├── problemstatement.txt
│   ├── PRD.md
│   └── architecture.md
├── src/mf_faq/
│   ├── config.py               # loads config.yaml + .env → Settings
│   ├── cli.py                  # `python -m mf_faq.cli <phase>`
│   ├── ingestion/
│   │   ├── loader.py           # P1 fetch → 01_raw
│   │   ├── cleaner.py          # P1 extract/normalize → 02_cleaned
│   │   ├── chunker.py          # P2 → 03_chunks
│   │   ├── embedder.py         # P3 → 04_embeddings
│   │   └── indexer.py          # P4 → 05_vectordb
│   ├── retrieval/
│   │   ├── embeddings.py       # shared embedding-model wrapper (ingest + query)
│   │   ├── vector_store.py     # ChromaDB wrapper
│   │   └── retriever.py        # P5 top-k, filters, threshold
│   ├── guardrails/
│   │   ├── pii.py              # regex detectors
│   │   ├── intent.py           # rules + LLM fallback classifier
│   │   └── validator.py        # post-generation checks
│   ├── generation/
│   │   ├── prompts.py          # system prompt, refusal templates
│   │   ├── llm/
│   │   │   ├── base.py         # LLMClient protocol
│   │   │   ├── groq_client.py
│   │   │   ├── gemini_client.py
│   │   │   ├── openai_client.py
│   │   │   ├── anthropic_client.py
│   │   │   ├── mistral_client.py
│   │   │   └── factory.py      # build client from Settings
│   │   └── answerer.py         # P6 orchestrates query pipeline
│   ├── formatting.py           # ≤3 sentences, citation, last-updated, disclaimer
│   └── logging_utils.py        # PII-safe retrieval logs → 06_retrieval_logs
├── app/
│   └── streamlit_app.py        # tiny UI
├── data/
│   ├── 01_raw/  02_cleaned/  03_chunks/  04_embeddings/  05_vectordb/  06_retrieval_logs/
└── tests/
    ├── test_pii.py  test_intent.py  test_chunker.py  test_validator.py  test_e2e_qa.py
```

## 5. Ingestion Pipeline

Each phase is a pure step from input artifact to output artifact, and can be run on its own:

```
python -m mf_faq.cli load      # P1a
python -m mf_faq.cli clean     # P1b
python -m mf_faq.cli chunk     # P2
python -m mf_faq.cli embed     # P3
python -m mf_faq.cli index     # P4
python -m mf_faq.cli ingest    # all of the above, in order
```

### 5.1 P1a: Loader
- Reads `sources.csv`: `url, slug, source_type, scheme_name, authority`.
  - `source_type` is one of `amc_overview | scheme | factsheet | how_to | education | regulation`.
- Fetching:
  - Uses `httpx` with a polite User-Agent, 1–2 seconds between requests, and checks robots.txt.
  - If the response has no usable content (for example the scheme fields are missing), retries with Playwright.
- Writes:
  - `01_raw/<slug>.html`
  - `01_raw/<slug>.meta.json` → `{url, slug, fetched_at, status, content_sha256, fetch_method}`
- Skips the network call when the content hash hasn't changed, unless `--force` is given.

### 5.2 P1b: Cleaner
- Groww scheme pages: parse the `__NEXT_DATA__` JSON and map known keys to a **fixed field set**. If the keys can't be found, fall back to CSS selectors.
- Narrative pages (AMFI, SEBI, RTA): strip navigation, footers, and scripts, and keep headings and paragraphs.
- **Drops returns/performance fields** (CAGR, 1Y/3Y/5Y returns, NAV history), as the PRD's "no performance claims" rule requires.
- Output `02_cleaned/<slug>.json`:

```json
{
  "slug": "hdfc-mid-cap-fund-direct-growth",
  "url": "https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth",
  "source_type": "scheme",
  "scheme_name": "HDFC Mid Cap Fund – Direct Growth",
  "amc": "HDFC Mutual Fund",
  "fetched_at": "2026-11-01T10:15:00+05:30",
  "fields": {
    "category": "...", "expense_ratio": "...", "exit_load": "...",
    "min_sip": "...", "min_lumpsum": "...", "lock_in": "...",
    "riskometer": "...", "benchmark": "...", "fund_manager": "...", "aum": "..."
  },
  "sections": [ { "heading": "...", "text": "..." } ],
  "factsheet_url": "..."
}
```

### 5.3 P2: Chunker (structure-aware)

| Content | Strategy | Example chunk text |
|---|---|---|
| `fields` on scheme pages | **Fact-card**: one chunk per field (or small related group), prefixed with scheme name | `HDFC Mid Cap Fund – Direct Growth — Exit load: 1% if redeemed within 1 year.` |
| `sections` (how-to, education, AMC overview) | Split by heading, then recursive split at `max_tokens=250`, `overlap=30` | `How to download capital-gains statement — Log in to …` |

- Tokens are counted with the `all-MiniLM-L6-v2` tokenizer. A hard limit of ≤ 256 tokens is checked by assertion.
- `chunk_id = sha1(slug + section + field + index)[:16]`. Because the ID is deterministic, re-ingestion upserts instead of duplicating.
- Output `03_chunks/chunks.jsonl`, one chunk per line:

```json
{"chunk_id":"a1b2c3d4e5f60718","text":"...","source_url":"...","slug":"...",
 "scheme_name":"...","amc":"HDFC Mutual Fund","doc_type":"scheme",
 "section":"exit_load","fetched_at":"2026-11-01T10:15:00+05:30","n_tokens":42}
```
- Also writes `03_chunks/stats.json` (counts per source and type, plus token-length histogram) for the demo.

### 5.4 P3: Embedder
- `retrieval/embeddings.py` wraps `SentenceTransformer(config.embedding.model_name)` with `normalize_embeddings=True`. The **same wrapper** is used at query time.
- Output:
  - `04_embeddings/embeddings.npy`, shape `(N, 384)`
  - `04_embeddings/ids.json`, chunk_ids in the same row order
  - `04_embeddings/manifest.json` → `{model_name, dim, normalized, created_at, n}`

### 5.5 P4: Indexer
- `chromadb.PersistentClient(path=config.vector_db.persist_dir)`.
- Collection `config.vector_db.collection` with metadata `{"hnsw:space": "cosine", "embedding_model": <model_name>}`.
- `collection.upsert(ids, embeddings, documents, metadatas)`, reading from the P2 and P3 artifacts.
- After indexing, checks that `collection.count() == len(chunks)`.

## 6. Query Pipeline

### 6.1 Sequence

```
UI ──q──► Answerer.answer(q)
           │
           ├─1─► PIIGuard.scan(q) ──hit──► Formatter.pii_rejection()           (q never logged)
           │
           ├─2─► IntentRouter.classify(q) ─► advice      ─► Formatter.refusal(edu_link)
           │                               ─► performance ─► Formatter.factsheet_redirect(scheme)
           │                               ─► out_of_scope ─► Formatter.not_available(closest_link)
           │                               ─► factual ↓
           │
           ├─3─► Retriever.search(q, scheme_filter?) ─► [chunks, scores]
           │         └─ max(score) < threshold ──► Formatter.not_available(...)
           │
           ├─4─► PromptBuilder.build(q, chunks) ─► LLMClient.complete(messages)
           │
           ├─5─► AnswerValidator.check(answer, chunks)
           │         ├─ cited URL ∈ retrieved URLs?    else → replace with top chunk URL
           │         ├─ ≤ 3 sentences?                 else → truncate
           │         └─ advice / return figures leaked? else → fall back to refusal
           │
           ├─6─► Formatter.answer(text, url, fetched_at)  → appends "Last updated from sources: <date>"
           │
           └─7─► RetrievalLogger.write(trace)  → data/06_retrieval_logs/<ts>.json
```

### 6.2 Components

| Component | Responsibility | Key interface |
|---|---|---|
| `PIIGuard` | Detects PAN, Aadhaar, phone, email, OTP, and account-number patterns | `scan(text) -> list[PIIMatch]` |
| `IntentRouter` | Rules first (keywords such as "should I", "best", "buy", "sell", "returns", "compare"); a short LLM classification for unclear cases | `classify(text) -> Intent` |
| `SchemeResolver` | Maps aliases ("mid cap", "ELSS", "tax saver", "flexi") to `scheme_name`, which becomes the metadata filter | `resolve(text) -> str \| None` |
| `Retriever` | Embeds the query, runs a Chroma query with `where` filter and `top_k`, applies the threshold | `search(q, scheme=None) -> list[ScoredChunk]` |
| `PromptBuilder` | System prompt plus numbered context blocks, each tagged with `[source_url]` | `build(q, chunks) -> list[Message]` |
| `LLMClient` | Calls the provider; temperature 0; timeout and retries | `complete(messages, **kw) -> str` |
| `AnswerValidator` | Enforces citation, length, and content rules after generation | `check(answer, chunks) -> ValidatedAnswer` |
| `Formatter` | Builds every user-facing response from templates | `answer() / refusal() / pii_rejection() / not_available()` |
| `RetrievalLogger` | Writes a PII-free trace of each query for the demo | `write(trace)` |

### 6.3 Scoring note
Chroma returns a cosine **distance**, which is converted with `similarity = 1 - distance`. The threshold check uses `config.retrieval.similarity_threshold` (default 0.35). Tune it on the labeled question set.

### 6.4 Prompt contract

System prompt (abridged):
```
You are a facts-only mutual fund FAQ assistant.
Rules:
- Answer ONLY using the CONTEXT. If the answer is not in CONTEXT, reply exactly: NOT_IN_SOURCES.
- Maximum 3 sentences. Plain, neutral language.
- Never give advice, opinions, recommendations, or compute/compare returns.
- End with: SOURCE: <one source_url copied from the CONTEXT block you used>.
```
User message: `CONTEXT:\n[1] (source_url=...) ...\n[2] ...\n\nQUESTION: <q>`

The validator parses `SOURCE:` from the reply. It maps `NOT_IN_SOURCES` to `Formatter.not_available`.

## 7. LLM Provider Abstraction

```python
class LLMClient(Protocol):
    name: str
    model: str
    def complete(self, messages: list[dict], temperature: float = 0.0,
                 max_tokens: int = 300) -> str: ...
```

- `factory.build_llm(settings)` reads `LLM_PROVIDER` and `LLM_MODEL` from `.env` and returns the matching adapter. The model ID is never hard-coded.
- Each adapter converts the shared `messages` format to its provider's SDK. For example, Anthropic and Gemini take the system prompt as a separate parameter.
- Startup check: if the API key for the selected provider is missing, or the model can't be reached, fail with a clear error.

| `LLM_PROVIDER` | Key variable | SDK |
|---|---|---|
| `groq` (default) | `GROQ_API_KEY` | `groq` |
| `gemini` | `GEMINI_API_KEY` | `google-genai` |
| `openai` | `OPENAI_API_KEY` | `openai` |
| `anthropic` | `ANTHROPIC_API_KEY` | `anthropic` |
| `mistral` | `MISTRAL_API_KEY` | `mistralai` |

## 8. Configuration

`config.py` loads `config.yaml` (non-secret) and `.env` (secrets plus LLM choice) into one typed `Settings` object (pydantic). The keys in `config.yaml` are as defined in PRD §7.1. One extra section is added for templates and links:

```yaml
links:
  education_default: <AMFI investor-education URL>
  statements_help:  <official RTA/AMC statement page URL>
ui:
  example_questions:
    - "What is the expense ratio of HDFC Mid Cap Fund?"
    - "What is the lock-in period of HDFC ELSS Tax Saver?"
    - "How do I download my capital-gains statement?"
```

**Consistency guard:** at query time, `Retriever` compares `config.embedding.model_name` with the `embedding_model` value stored in the collection's metadata. If they differ, it refuses to run and prints a "re-run ingest" message.

## 9. Guardrails: Where They Sit

| Stage | Guardrail | Failure action |
|---|---|---|
| Before anything | `PIIGuard` | PII rejection message. Raw text is not logged or sent to the LLM. |
| Before retrieval | `IntentRouter` | Refusal with education link, or factsheet redirect |
| After retrieval | Similarity threshold | "Not available in sources" with the closest official link |
| In prompt | System rules, context-only, `NOT_IN_SOURCES` sentinel | — |
| After generation | `AnswerValidator`: citation in retrieved set, ≤ 3 sentences, no advice or return numbers (regex for `%` near "return/CAGR") | Fix the citation, truncate, or fall back to refusal |
| Ingestion | Cleaner drops performance fields | Return data never enters the index |
| Logging | `RetrievalLogger` stores only PII-free queries; PII hits are recorded as `{"event":"pii_rejected","types":[...]}` | — |

PII patterns (in `guardrails/pii.py`):

| Type | Pattern (simplified) |
|---|---|
| PAN | `\b[A-Z]{5}[0-9]{4}[A-Z]\b` |
| Aadhaar | `\b\d{4}\s?\d{4}\s?\d{4}\b` |
| Phone (IN) | `\b(\+91[\s-]?)?[6-9]\d{9}\b` |
| Email | `\b[\w.+-]+@[\w-]+\.[\w.]+\b` |
| OTP | `\b(otp\|one[- ]time)\b.*\b\d{4,8}\b` (case-insensitive) |
| Account no. | `\b\d{9,18}\b` |

## 10. UI (Streamlit)

```
┌──────────────────────────────────────────────────────────┐
│  HDFC Mutual Fund FAQ Assistant                          │
│  Hi! Ask me factual questions about HDFC MF schemes.     │
│  ⚠ Facts-only. No investment advice.                    │
│                                                          │
│  [Expense ratio of HDFC Mid Cap?] [ELSS lock-in?]        │
│  [How to download capital-gains statement?]              │
│                                                          │
│  ┌─ chat ───────────────────────────────────────────┐    │
│  │ Q: ...                                           │    │
│  │ A: ... (≤3 sentences)                            │    │
│  │    Source: <link>                                │    │
│  │    Last updated from sources: 01 Nov 2026        │    │
│  └──────────────────────────────────────────────────┘    │
│  [ Type your question…                        ] [Send]   │
│                                                          │
│  ▸ Show pipeline (intent, retrieved chunks + scores,     │
│    metadata, prompt, raw LLM output, validator result)   │
│  ─ Disclaimer snippet (footer)                           │
└──────────────────────────────────────────────────────────┘
```
- The UI never shows the user's input again after a PII rejection.
- The "Show pipeline" panel reads the same trace object that `RetrievalLogger` writes.
- A sidebar, for the demo only, shows the counts from `03_chunks/stats.json` and `04_embeddings/manifest.json`.

## 11. Core Data Models

```python
class Chunk(BaseModel):
    chunk_id: str; text: str; source_url: str; slug: str
    scheme_name: str | None; amc: str; doc_type: str
    section: str; fetched_at: datetime; n_tokens: int

class ScoredChunk(BaseModel):
    chunk: Chunk; similarity: float

class Intent(str, Enum):
    FACTUAL = "factual"; ADVICE = "advice"; PERFORMANCE = "performance"
    PII = "pii"; OUT_OF_SCOPE = "out_of_scope"

class AnswerResponse(BaseModel):
    intent: Intent
    text: str                  # ≤ 3 sentences
    citation_url: str          # exactly one
    last_updated: date | None
    trace_id: str
```

## 12. Error Handling & Non-Functional Requirements

| Concern | Approach |
|---|---|
| LLM timeout or error | 2 retries with backoff, then "Sorry, I couldn't answer right now", plus the top-chunk link |
| Scrape failure | Phase reports which URLs failed; the existing artifacts are kept |
| Latency target | p50 under 3 s per query (embedding on CPU takes about 20 ms; the LLM call dominates) |
| Determinism | Temperature 0, fixed `top_k`, deterministic chunk IDs |
| Secrets | Only in `.env` (git-ignored). Never logged. |
| Observability | `06_retrieval_logs/*.json` holds intent, scheme filter, chunk IDs, scores, prompt hash, latency, provider, and model |

## 13. Testing Strategy

| Test | Covers |
|---|---|
| `test_pii.py` | Each PII type is detected, and harmless numbers ("₹500 SIP", "1 year") are not flagged |
| `test_intent.py` | Advice and performance prompts are routed correctly |
| `test_chunker.py` | Token limit is respected, metadata is complete, IDs stay the same across runs |
| `test_validator.py` | Out-of-context URLs are replaced, long answers are truncated, return figures are blocked |
| `test_e2e_qa.py` | A labeled set of 20+ questions (every scheme × key field), checked against PRD §10 metrics; this run also produces `sample_qa.md` |

## 14. Deployment

- **Local:**
  1. `pip install -r requirements.txt`
  2. `playwright install chromium` (only needed for the fallback)
  3. `cp .env.example .env`
  4. `python -m mf_faq.cli ingest`
  5. `streamlit run app/streamlit_app.py`
- **Hosted prototype (D1):** Streamlit Community Cloud or Hugging Face Spaces. Commit `data/05_vectordb` (it is small) so the hosted app doesn't need to scrape, and set API keys as platform secrets.

## 15. Mapping PRD → Architecture

| PRD item | Implemented by |
|---|---|
| FR-1 classification | `guardrails/intent.py`, `guardrails/pii.py` |
| FR-2/3/4 answer, citation, last-updated | `generation/answerer.py`, `guardrails/validator.py`, `formatting.py` |
| FR-5/6 refusal and factsheet redirect | `formatting.py`, `generation/prompts.py`, `links` config |
| FR-7 PII | `guardrails/pii.py`, `logging_utils.py` |
| FR-8/9 not-in-sources and threshold | `retrieval/retriever.py`, `formatting.py` |
| UI-1…5 | `app/streamlit_app.py` |
| §6 phases, per-stage artifacts | `ingestion/*`, `cli.py`, `data/01…06` |
| §7 config and `.env` | `config.py`, `config.yaml`, `generation/llm/factory.py` |
| §8 guardrails | §9 of this doc |
| §9 deliverables | `sources.csv`, `README.md`, `sample_qa.md`, test e2e run |
