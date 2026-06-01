"""Cliente de chat (LLM), agnóstico del proveedor (OpenAI u Ollama).

El chat de Ollama es estable vía su endpoint compatible con OpenAI (`/v1`)
(a diferencia de los embeddings, por eso allí se usa el endpoint nativo).
"""
from openai import OpenAI

from .config import settings


def _client() -> OpenAI:
    if settings.chat_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("Falta OPENAI_API_KEY en el .env (LLM_PROVIDER=openai).")
        kwargs = {"api_key": settings.openai_api_key, "timeout": settings.request_timeout}
        if settings.openai_base_url:  # endpoint compatible (Gemini, Groq, etc.)
            kwargs["base_url"] = settings.openai_base_url
        return OpenAI(**kwargs)
    return OpenAI(
        base_url=f"{settings.ollama_host}/v1",
        api_key="ollama",  # ignorada por Ollama, pero requerida por el cliente
        timeout=settings.request_timeout,
    )


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Envía los mensajes al LLM y devuelve el texto de la respuesta."""
    resp = _client().chat.completions.create(
        model=settings.chat_model,
        messages=messages,
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()
