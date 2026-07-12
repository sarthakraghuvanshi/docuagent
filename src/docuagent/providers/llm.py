"""LLM provider behind a single OpenAI-compatible interface.

Ollama, Vertex, and OpenAI all speak (or can be adapted to) this shape, so
switching providers is a config change, not a rewrite.
"""

from typing import Protocol

import httpx

from docuagent.config import settings


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str: ...


class OllamaLLM:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, system: str, user: str, *, temperature: float = 0.0) -> str:
        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def is_healthy(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


def get_llm() -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaLLM(settings.ollama_base_url, settings.llm_model)
    raise NotImplementedError(
        f"LLM provider '{settings.llm_provider}' not wired yet (Phase 7 adds vertex/openai)."
    )
