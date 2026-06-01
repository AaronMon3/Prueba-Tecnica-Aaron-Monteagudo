"""API REST (FastAPI) — endpoint que consume el workflow de n8n.

/query siempre responde 200 con un JSON {encontrado, respuesta, fuentes},
incluso ante input vacío o errores internos (se loguean server-side).
"""
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from .config import settings
from .rag import answer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("minecatalog.api")

app = FastAPI(title="Asistente de soporte MineCatalog")

EMPTY_MSG = "La consulta está vacía. Por favor, escribí tu pregunta."
ERROR_MSG = (
    "Ocurrió un problema técnico al procesar tu consulta. "
    "Por favor, intentá nuevamente o contactá al soporte (soporte.minecatalog@empresa.com)."
)


class Query(BaseModel):
    pregunta: str = ""  # opcional: se valida manualmente para responder con elegancia


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "chat": {"provider": settings.chat_provider, "model": settings.chat_model},
        "embeddings": {"provider": settings.embeddings_provider, "model": settings.embedding_model},
    }


@app.post("/query")
def query(q: Query) -> dict:
    pregunta = (q.pregunta or "").strip()
    if not pregunta:
        return {"encontrado": False, "respuesta": EMPTY_MSG, "fuentes": []}
    try:
        return answer(pregunta)
    except Exception:
        logger.exception("Error al procesar la consulta: %r", pregunta)
        return {"encontrado": False, "respuesta": ERROR_MSG, "fuentes": []}
