# PRD: Groww Mutual Fund Facts (RAG Demo)

| Field | Value |
|---|---|
| Product | Groww Mutual Fund Facts |
| Source | `docs/problemstatement.txt` |
| Scope | HDFC Mutual Fund (AMC) + 4 schemes, sourced from Groww public pages |
| Status | Draft v1.0 |
| Date | 2026-11-01 |

---

## 1. Overview

We are building a small FAQ assistant that answers **factual** questions about selected HDFC mutual fund schemes, such as expense ratio, exit load, minimum SIP, ELSS lock-in, riskometer, benchmark, and how to download statements. It uses only official public pages.

Every answer is grounded in retrieved source text and cites **exactly one source link**. The assistant never gives investment advice.

The project is also a **teaching demo of Retrieval-Augmented Generation (RAG)**. Each pipeline stage (loading, chunking, embedding, storage, retrieval, generation) saves its output to disk so it can be inspected and shown during the demo. The build proceeds phase by phase.

## 2. Problem & Users

**Problem:** Retail investors and support teams spend time looking up the same scheme facts on several pages. General-purpose chatbots can make up numbers or slip into advice. Users need short, correct, cited answers with no opinions.

| User | Need |
|---|---|
| Retail investor comparing schemes | Quick, trustworthy facts (expense ratio, exit load, lock-in, SIP minimum) with a link to verify |
| Support / content team | Consistent, citable answers to repetitive MF questions |
| Demo audience / evaluators | Visibility into each RAG stage and how the answer was produced |

## 3. Goals & Non-Goals

### Goals
- G1: Answer factual scheme/AMC questions correctly from the stored corpus.
- G2: Include exactly one clear citation URL in every answer.
- G3: Politely refuse advice, opinion, portfolio, and return-comparison questions, and give one relevant educational link.
- G4: Never accept or store PII.
- G5: Save each RAG stage's output as inspectable files.
- G6: Make the embedding model, vector DB, LLM provider, and model configurable without code changes.

### Non-Goals
- Buy/sell/hold advice, recommendations, or suitability checks.
- Computing, comparing, or projecting returns or performance.
- User accounts, transactions, or portfolio access.
- Coverage beyond the defined AMC and schemes (v1).
- Using third-party blogs, forums, or app back-end screenshots as sources.

## 4. Scope: Data Sources

### 4.1 Primary corpus (required)

| # | Type | Page | URL |
|---|---|---|---|
| 1 | AMC overview | HDFC Mutual Fund | https://groww.in/mutual-funds/amc/hdfc-mutual-funds |
| 2 | Large Cap | HDFC Large Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-large-cap-fund-direct-growth |
| 3 | Flexi Cap | HDFC Flexi Cap – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-equity-fund-direct-growth |
| 4 | ELSS | HDFC ELSS Tax Saver – Direct Plan Growth | https://groww.in/mutual-funds/hdfc-elss-tax-saver-fund-direct-plan-growth |
| 5 | Mid Cap | HDFC Mid Cap Fund – Direct Growth | https://groww.in/mutual-funds/hdfc-mid-cap-fund-direct-growth |

### 4.2 Supplementary corpus (to reach 15–25 URLs)
The deliverable needs a source list of 15–25 URLs. Additional pages must be **official public pages** from the AMC, SEBI, AMFI, or the registrars (RTAs). Examples:
- HDFC MF official scheme pages, factsheets, SID/KIM documents (for factsheet links on performance questions).
- HDFC MF / CAMS / KFintech pages on downloading account and capital-gains statements.
- AMFI investor education pages (riskometer, SIP, ELSS basics). These also serve as refusal links.
- SEBI investor pages and circulars on the riskometer and on scheme categorization.

> Every supplementary URL must be checked by hand before it goes into `sources.csv`. No third-party blogs.

## 5. Functional Requirements

### 5.1 Question handling

| ID | Requirement |
|---|---|
| FR-1 | Classify each query into one of five classes: **factual**, **advice/opinion**, **performance/returns**, **PII-containing**, or **out-of-scope**. |
| FR-2 | **Factual:** retrieve relevant chunks and generate an answer of **≤ 3 sentences**, using only retrieved context. |
| FR-3 | Each answer includes **exactly one** citation link: the source URL of the chunk the answer comes from. |
| FR-4 | Each answer ends with `Last updated from sources: <date>`, where the date is the scrape timestamp of the cited source. |
| FR-5 | **Advice/opinion** (for example "Should I buy/sell X?" or "Which fund is best?"): give a polite facts-only refusal plus one educational link (AMFI/SEBI). |
| FR-6 | **Performance/returns** (for example "What returns did X give?" or "Compare returns"): do not compute or compare. Point the user to the official factsheet link. |
| FR-7 | **PII detected** (PAN, Aadhaar, account number, OTP, email, phone): reject the message, do not process it further, and **do not log or store** the raw text. Ask the user to remove personal details. |
| FR-8 | **Out-of-scope or not in corpus:** say the information is not available in the sources and point to the closest relevant official page. Never guess. |
| FR-9 | If retrieval confidence is below the configured similarity threshold, treat the query as FR-8 rather than generating an answer. |

### 5.2 Example supported intents
- "What is the expense ratio of HDFC Mid Cap Fund?"
- "What is the lock-in period for HDFC ELSS Tax Saver?"
- "Minimum SIP amount for HDFC Large Cap Fund?"
- "Exit load on HDFC Flexi Cap?"
- "What is the riskometer level / benchmark of HDFC Mid Cap?"
- "How do I download my capital-gains statement?"

### 5.3 UI (tiny)

| ID | Requirement |
|---|---|
| UI-1 | Welcome line introducing the assistant. |
| UI-2 | Three clickable example questions. |
| UI-3 | A visible note: **"Facts-only. No investment advice."** |
| UI-4 | Chat input and answer area. The citation shows as a clickable link. |
| UI-5 | *(Demo)* An optional "Show pipeline" panel listing the retrieved chunks, similarity scores, metadata, and final prompt for the last answer. |

## 6. RAG Pipeline: Phase-wise Design

Every phase writes its output under `data/` so it can be inspected on its own and re-run independently.

```
data/
├── 01_raw/          # raw HTML/JSON per URL + fetch metadata
├── 02_cleaned/      # extracted, normalized text/structured fields per page
├── 03_chunks/       # chunks.jsonl (text + metadata)
├── 04_embeddings/   # embeddings.npy / embeddings.jsonl (chunk_id → vector)
├── 05_vectordb/     # persisted ChromaDB directory
└── 06_retrieval_logs/  # per-query: retrieved chunks, scores, prompt, answer (PII-free)
```

### Phase 1: Data Loading (Scraping)
- Input: `sources.csv` (url, source_type, scheme_name, authority).
- Fetch each page. Groww pages are rendered by JavaScript, so either parse the embedded page JSON (for example `__NEXT_DATA__`) or use a headless browser (Playwright) as a fallback.
- Save raw output to `01_raw/<slug>.html` together with `<slug>.meta.json` (url, fetched_at, HTTP status, content hash).
- Extract and clean text into `02_cleaned/<slug>.json`, keeping both the **structured fields** (expense ratio, exit load, min SIP, lock-in, riskometer, benchmark, fund manager, AUM, category) and the **narrative text**.
- **Remove returns/performance figures** from the indexed corpus so the bot cannot cite them as claims. Keep the factsheet link.

### Phase 2: Data Chunking
The strategy follows the data. The coding agent may refine it, but must document the reasons in the README.
- **Structured scheme facts → one "fact-card" chunk per field group.** Example: `"HDFC Mid Cap Fund – Direct Growth | Exit load: 1% if redeemed within 1 year…"`. Each chunk starts with the scheme name so it matches on its own without nearby context.
- **Narrative / FAQ / how-to pages → split at headings first, then by size.** Use recursive splitting at about 150–250 tokens with about 30 tokens of overlap. This keeps chunks within the **256-token input limit of all-MiniLM-L6-v2**.
- Metadata on every chunk: `chunk_id, source_url, scheme_name, amc, doc_type, section, fetched_at`.
- Output: `03_chunks/chunks.jsonl` plus a summary of chunk counts and lengths.

### Phase 3: Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim), with normalized embeddings and cosine similarity.
- Output: `04_embeddings/` with vectors keyed by `chunk_id`, plus the model name and version.

### Phase 4: Vector Store
- **ChromaDB** with a persistent client at `05_vectordb/`. Collection name comes from config.
- Store documents, embeddings, and full metadata. Ingestion must be idempotent (re-running upserts by `chunk_id`).

### Phase 5: Retrieval
- Embed the query with the **same embedding model** used at ingestion. This is enforced by checking the model name stored in collection metadata.
- Run a top-k similarity search, with a metadata filter by scheme when the query names one.
- Apply the similarity threshold (FR-9). Log the results to `06_retrieval_logs/`.

### Phase 6: Generation
- The LLM receives a system prompt plus the retrieved chunks, and must follow these rules:
  - answer only from the given context, in ≤ 3 sentences;
  - give no advice, opinions, or return calculations;
  - cite one `source_url` from the context;
  - say "not available in sources" when the context doesn't contain the answer.
- Post-processing:
  - enforce the sentence limit;
  - check that the cited URL is one of the retrieved sources;
  - append `Last updated from sources: <date>`.

## 7. Configuration

### 7.1 `config.yaml` (non-secret, changeable)
```yaml
embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  normalize: true
vector_db:
  provider: chromadb
  persist_dir: data/05_vectordb
  collection: hdfc_mf_faq
chunking:
  strategy: structure_aware
  max_tokens: 250
  overlap_tokens: 30
retrieval:
  top_k: 4
  similarity_threshold: 0.35
generation:
  max_sentences: 3
  temperature: 0.0
```

### 7.2 `.env` (secrets + LLM selection)
```
LLM_PROVIDER=groq            # groq | gemini | openai | anthropic | mistral
LLM_MODEL=qwen/qwen3.8-27b   # read from env, never hard-coded
GROQ_API_KEY=
GEMINI_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
MISTRAL_API_KEY=
```
- Groq is the default provider. A small provider-adapter interface lets you switch to Gemini, OpenAI, Anthropic, or Mistral by changing `.env` only.
- `.env` is git-ignored. Ship `.env.example` instead.

## 8. Guardrails & Compliance

| Area | Rule |
|---|---|
| Sources | Public, official pages only (Groww scheme pages as scoped, AMC, SEBI, AMFI, RTAs). No blogs. No back-end screenshots. |
| PII | Pre-LLM regex/validator checks for PAN (`[A-Z]{5}[0-9]{4}[A-Z]`), Aadhaar (12 digits), phone (10 digits starting 6–9), email, OTP patterns, and account-number-like digit strings. Matching messages are rejected and never written to logs. |
| Advice | Intent classification (rules + LLM) catches buy/sell/hold/"best fund"/"should I" questions and returns a refusal with an educational link. |
| Performance | No return figures in the corpus, and no computing or comparing returns. Link to the official factsheet. |
| Transparency | Answers ≤ 3 sentences, one citation, last-updated date, disclaimer always shown. |

**Disclaimer snippet (UI):**
> Facts-only. No investment advice. Answers are drawn from official public sources and may not reflect the latest updates. Please verify on the linked page. Mutual fund investments are subject to market risks; read all scheme-related documents carefully.

**Refusal template:**
> I can only share factual information about mutual fund schemes and can't offer investment advice. For guidance on choosing funds, see this investor-education resource: <AMFI/SEBI link>.

## 9. Deliverables

| # | Deliverable | Format |
|---|---|---|
| D1 | Working prototype (app/notebook link), or a demo video of ≤ 3 minutes | URL / video |
| D2 | Source list of 15–25 URLs | `sources.csv` or `sources.md` |
| D3 | README covering setup steps, scope (AMC + schemes), chunking rationale, and known limits | `README.md` |
| D4 | Sample Q&A of 5–10 queries with answers and links, including at least 2 refusals and 1 PII rejection | `sample_qa.md` |
| D5 | Disclaimer snippet used in the UI | In README + UI |
| D6 | Per-stage data artifacts | `data/01_raw` … `data/06_retrieval_logs` |

## 10. Success Metrics / Acceptance Criteria

- 100% of factual answers contain exactly one valid citation URL taken from the retrieved sources.
- 100% of answers are ≤ 3 sentences and include `Last updated from sources:`.
- 100% of the advice and performance test prompts are refused or redirected correctly.
- 100% of the PII test inputs are rejected and none appear in logs.
- ≥ 90% factual accuracy on a hand-labeled set of 20+ questions covering every scheme × key field.
- Each pipeline phase can be run on its own and writes the artifacts listed above.
- Switching `LLM_PROVIDER`/`LLM_MODEL` in `.env` works without code changes.

## 11. Implementation Phases (Milestones)

| Phase | Output | Exit criteria |
|---|---|---|
| P0 Setup | Repo structure, `config.yaml`, `.env.example`, `sources.csv` | Config loads; URLs verified |
| P1 Loading | `01_raw`, `02_cleaned` | All sources fetched; key fields extracted for 4 schemes |
| P2 Chunking | `03_chunks/chunks.jsonl` | Every chunk ≤ 250 tokens, with full metadata |
| P3 Embedding | `04_embeddings/` | One vector per chunk, 384-dim |
| P4 Vector DB | `05_vectordb/` | Collection count = chunk count; re-run is idempotent |
| P5 Retrieval | Retrieval module + logs | Correct chunk in top-k for the labeled set |
| P6 Generation + Guardrails | Answer pipeline, PII/advice filters | Section 10 criteria met |
| P7 UI + Deliverables | Tiny UI, README, sample Q&A, demo | All of D1–D6 complete |

## 12. Risks & Known Limits

- **JS-rendered or changing pages:** Groww's layout may change and break extraction. Mitigation: parse the embedded JSON, use a Playwright fallback, and validate the extracted fields.
- **Stale data:** facts reflect the scrape date only. This is shown to the user through "Last updated from sources".
- **Small embedding model:** all-MiniLM-L6-v2 truncates input at 256 tokens and may miss subtle wording. This is handled by small, self-contained chunks.
- **Distributor vs. official source:** Groww is a distributor platform. Where the problem statement requires AMC/SEBI/AMFI sources, prefer those for citations.
- **Scraping terms:** follow the sites' robots.txt and terms, and scrape at a low request rate.
- **Model availability:** the configured Groq model ID must exist at runtime. Check it at startup and fail with a clear error if it doesn't.

## 13. Open Questions

1. Should the answer cite the official AMC/AMFI/SEBI page when the same fact also appears on Groww, or always the Groww page?
2. Which supplementary official URLs (factsheets, SID/KIM, RTA statement pages) should make up the 15–25 source list?
3. Which UI framework should the prototype use (Streamlit, Gradio, or a notebook)? *Suggested: Streamlit, for fast hosting and an easy pipeline panel.*
4. How often should the corpus be refreshed (manual re-run, or scheduled)?
