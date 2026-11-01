"""LLM providers. LLM_PROVIDER and LLM_MODEL come from .env (Groq is the default).

Every client exposes: complete(messages, temperature, max_tokens) -> str
`messages` uses the OpenAI-style format: [{"role": "system"|"user"|"assistant", "content": "..."}]
"""
import re
from functools import lru_cache

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


class GroqClient:
    name = "groq"

    def __init__(self, api_key: str, model: str, timeout: float):
        from groq import Groq
        self.client, self.model = Groq(api_key=api_key, timeout=timeout), model

    def complete(self, messages, temperature=0.0, max_tokens=1024) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                temperature=temperature, max_tokens=max_tokens)
        return clean_output(r.choices[0].message.content)


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str, model: str, timeout: float):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key, timeout=timeout), model

    def complete(self, messages, temperature=0.0, max_tokens=1024) -> str:
        r = self.client.chat.completions.create(model=self.model, messages=messages,
                                                temperature=temperature, max_completion_tokens=max_tokens)
        return clean_output(r.choices[0].message.content)


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str, model: str, timeout: float):
        import anthropic
        self.client, self.model = anthropic.Anthropic(api_key=api_key, timeout=timeout), model

    def complete(self, messages, temperature=0.0, max_tokens=1024) -> str:
        system, msgs = split_system(messages)
        r = self.client.messages.create(model=self.model, system=system, messages=msgs,
                                        temperature=temperature, max_tokens=max_tokens)
        return clean_output("".join(b.text for b in r.content if getattr(b, "type", "") == "text"))


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float):
        from google import genai
        self.client, self.model = genai.Client(api_key=api_key), model

    def complete(self, messages, temperature=0.0, max_tokens=1024) -> str:
        from google.genai import types
        system, msgs = split_system(messages)
        contents = [types.Content(role="user" if m["role"] == "user" else "model",
                                  parts=[types.Part(text=m["content"])]) for m in msgs]
        r = self.client.models.generate_content(
            model=self.model, contents=contents,
            config=types.GenerateContentConfig(system_instruction=system or None,
                                               temperature=temperature, max_output_tokens=max_tokens))
        return clean_output(r.text)


class MistralClient:
    name = "mistral"

    def __init__(self, api_key: str, model: str, timeout: float):
        try:
            from mistralai import Mistral
        except ImportError:                       # newer SDK layout
            from mistralai.client import Mistral
        self.client, self.model = Mistral(api_key=api_key, timeout_ms=int(timeout * 1000)), model

    def complete(self, messages, temperature=0.0, max_tokens=1024) -> str:
        r = self.client.chat.complete(model=self.model, messages=messages,
                                      temperature=temperature, max_tokens=max_tokens)
        return clean_output(r.choices[0].message.content)


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
