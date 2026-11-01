"""Load settings from config.yaml (non-secret) and .env (secrets + LLM choice)."""
import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ("groq", "gemini", "openai", "anthropic", "mistral")
REQUIRED_SECTIONS = ("scraping", "embedding", "vector_db", "chunking", "retrieval", "generation", "guardrails", "links", "ui")


@lru_cache
def get_config() -> dict:
    """Return config.yaml as a dict, checking that every section is present."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(f"config.yaml is missing sections: {missing}")
    return cfg


@lru_cache
def get_env() -> dict:
    """Return the LLM provider, model and that provider's API key from .env."""
    load_dotenv(ROOT / ".env")
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    if provider not in PROVIDERS:
        raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got '{provider}'")
    model = os.getenv("LLM_MODEL", "").strip()
    if not model:
        raise ValueError("LLM_MODEL is not set in .env")
    return {
        "provider": provider,
        "model": model,
        "api_key": os.getenv(f"{provider.upper()}_API_KEY", "").strip() or None,
    }


def path(rel: str) -> Path:
    """Resolve a project-relative path, e.g. path('data/01_raw')."""
    return ROOT / rel
