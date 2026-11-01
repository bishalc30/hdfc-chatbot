# HDFC Mutual Fund FAQ Assistant (RAG Demo)

A facts-only FAQ assistant for HDFC Mutual Fund schemes, built as a step-by-step RAG demo.
See `docs/PRD.md`, `docs/architecture.md` and `docs/implementation.md`.

> Facts-only. No investment advice.

## Setup

1. Install Python 3.11+ from https://www.python.org/downloads/ (tick "Add python.exe to PATH").
2. Create a virtual environment and install dependencies:
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` (already done locally) and set the API key for `LLM_PROVIDER`.
4. Check the configuration loads:
   ```powershell
   python -c "from src.config import get_config, get_env; print(get_config()['embedding']); print(get_env()['provider'])"
   ```

## Pipeline steps

The `data/` folders are not included in the repository. Build them locally before starting the app:

```powershell
python -m src.pipeline scrape   # step 2: fetch pages      -> data/01_raw  (add --force to re-save unchanged pages)
python -m src.pipeline load     # step 3: parse + clean    -> data/02_cleaned
python -m src.pipeline chunk    # step 4: chunking         -> data/03_chunks
python -m src.pipeline embed    # step 5: embeddings       -> data/04_embeddings
python -m src.pipeline index    # step 6: ChromaDB         -> data/05_vectordb
python -m src.pipeline all      # all of the above
```

## Scope, known limits, disclaimer

_To be completed in later steps._
