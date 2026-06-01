"""Generación de embeddings, agnóstica del proveedor (OpenAI u Ollama).

- OpenAI: cliente oficial.
- Ollama: endpoint nativo `/api/embed`, más fiable para embeddings que el
  endpoint compatible `/v1` (este último puede devolver NaN en batch con bge-m3).
"""
import requests
from openai import OpenAI

from .config import settings


def _embed_openai(texts: list[str]) -> list[list[float]]:
    if not settings.openai_api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en el .env (LLM_PROVIDER=openai).")
    kwargs = {"api_key": settings.openai_api_key}
    if settings.openai_base_url:  # endpoint compatible (Gemini, Groq, etc.)
        kwargs["base_url"] = settings.openai_base_url
    client = OpenAI(**kwargs)
    resp = client.embeddings.create(model=settings.openai_embedding_model, input=texts)
    return [item.embedding for item in resp.data]


def _embed_ollama(texts: list[str]) -> list[list[float]]:
    url = f"{settings.ollama_host}/api/embed"
    resp = requests.post(
        url,
        json={"model": settings.ollama_embedding_model, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Devuelve un embedding por cada texto, en el mismo orden."""
    if not texts:
        return []
    if settings.embeddings_provider == "openai":
        return _embed_openai(texts)
    return _embed_ollama(texts)


def embed_text(text: str) -> list[float]:
    """Embedding de un único texto."""
    return embed_texts([text])[0]
