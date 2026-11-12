"""LLM providers. LLM_PROVIDER and LLM_MODEL come from .env (Groq is the default).

Every client exposes: complete(messages, temperature, max_tokens, timeout=None) -> str
(timeout: per-call seconds; defaults to generation.timeout_s)
`messages` uses the OpenAI-style format: [{"role": "system"|"user"|"assistant", "content": "..."}]

All providers are called through their public HTTP APIs with httpx, so no provider SDKs are
needed (this keeps the Vercel function small). Switch provider by changing .env only.
"""
import re
from functools import lru_cache

import httpx

from src.config import get_config, get_env

THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def clean_output(text: str | None) -> str:
    """Remove reasoning blocks (e.g. Qwen3 <think>...</think>) and surrounding whitespace."""
    text = THINK.sub("", text or "")
    if "<think>" in text.lower():                 # unterminated block (output cut off while thinking)
        text = text[: text.lower().index("<think>")]
    return text.strip()


def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


class LLMError(RuntimeError):
    pass


class _HTTPClient:
    name = "base"

    def __init__(self, api_key: str, model: str, timeout: float):
        self.api_key, self.model = api_key, model
        self.http = httpx.Client(timeout=timeout)

    def _post(self, url: str, headers: dict, body: dict, timeout: float | None = None) -> dict:
        r = self.http.post(url, headers={"Content-Type": "application/json", **headers}, json=body,
                           timeout=timeout or self.http.timeout)
        if r.status_code >= 400:
            # Provider error text only (never the request, which holds the prompt/key headers)
            raise LLMError(f"{self.name} API {r.status_code}: {r.text[:300]}")
        return r.json()


class OpenAICompatibleClient(_HTTPClient):
    """Chat Completions format: Groq, OpenAI and Mistral."""
    url = ""
    tokens_field = "max_tokens"

    def complete(self, messages, temperature=0.0, max_tokens=1024, timeout=None) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                self.tokens_field: max_tokens}
        data = self._post(self.url, {"Authorization": f"Bearer {self.api_key}"}, body, timeout)
        return clean_output(data["choices"][0]["message"].get("content"))


class GroqClient(OpenAICompatibleClient):
    name = "groq"
    url = "https://api.groq.com/openai/v1/chat/completions"


class OpenAIClient(OpenAICompatibleClient):
    name = "openai"
    url = "https://api.openai.com/v1/chat/completions"
    tokens_field = "max_completion_tokens"


class MistralClient(OpenAICompatibleClient):
    name = "mistral"
    url = "https://api.mistral.ai/v1/chat/completions"


class AnthropicClient(_HTTPClient):
    name = "anthropic"
    url = "https://api.anthropic.com/v1/messages"

    def complete(self, messages, temperature=0.0, max_tokens=1024, timeout=None) -> str:
        system, msgs = split_system(messages)
        body = {"model": self.model, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
        if system:
            body["system"] = system
        data = self._post(self.url, {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, body, timeout)
        return clean_output("".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"))


class GeminiClient(_HTTPClient):
    name = "gemini"

    def complete(self, messages, temperature=0.0, max_tokens=1024, timeout=None) -> str:
        system, msgs = split_system(messages)
        body = {
            "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]}
                         for m in msgs],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        data = self._post(url, {"x-goog-api-key": self.api_key}, body, timeout)
        parts = (data.get("candidates") or [{}])[0].get("content", {}).get("parts", [])
        return clean_output("".join(p.get("text", "") for p in parts))


CLIENTS = {"groq": GroqClient, "openai": OpenAIClient, "anthropic": AnthropicClient,
           "gemini": GeminiClient, "mistral": MistralClient}


@lru_cache
def get_llm():
    env = get_env()
    if not env["api_key"]:
        raise RuntimeError(f"{env['provider'].upper()}_API_KEY is empty in .env "
                           f"(LLM_PROVIDER={env['provider']}).")
    timeout = get_config()["generation"].get("timeout_s", 30)
    return CLIENTS[env["provider"]](env["api_key"], env["model"], timeout)
