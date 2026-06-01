"""Recuperación + generación.

Busca los fragmentos más relevantes a la pregunta (búsqueda semántica),
arma el contexto y pide al LLM una respuesta basada únicamente en ese
contexto, con citación de fuentes.
"""
import atexit

from .config import settings
from .embeddings import embed_text
from .llm import chat
from .vectorstore import VectorStore

NO_ANSWER = (
    "No encontré esa información en la documentación disponible. "
    "Te sugiero contactar al soporte técnico (soporte.minecatalog@empresa.com)."
)

SYSTEM_PROMPT = (
    "Eres un asistente de soporte técnico del software MineCatalog.\n"
    "Respondes ÚNICAMENTE con la información del CONTEXTO que se te proporciona.\n\n"
    "Reglas:\n"
    f'- Si el CONTEXTO no contiene la respuesta, responde exactamente: "{NO_ANSWER}"\n'
    "- No inventes información ni uses conocimiento externo.\n"
    "- Responde en español, de forma clara y concisa.\n"
    "- Al final de cada afirmación relevante cita la fuente entre corchetes, "
    "por ejemplo: [Documentación 2.txt · 3.2]."
)

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Abre Qdrant una sola vez y reutiliza la conexión durante el proceso."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


@atexit.register
def _close_store() -> None:
    if _store is not None:
        try:
            _store.close()
        except Exception:
            pass


def _build_context(hits) -> str:
    bloques = []
    for h in hits:
        p = h.payload
        fuente = p["source"] + (f" · {p['section']}" if p.get("section") else "")
        bloques.append(f"[Fuente: {fuente}]\n{p['text']}")
    return "\n\n---\n\n".join(bloques)


def answer(question: str) -> dict:
    """Responde una pregunta. Devuelve {encontrado, respuesta, fuentes}."""
    question = (question or "").strip()
    if not question:
        return {"encontrado": False, "respuesta": "La pregunta está vacía.", "fuentes": []}

    hits = _get_store().search(embed_text(question), top_k=settings.top_k)
    relevantes = [h for h in hits if h.score >= settings.score_threshold]

    # Filtro grueso: si nada supera el umbral, no se gasta el LLM.
    if not relevantes:
        return {"encontrado": False, "respuesta": NO_ANSWER, "fuentes": []}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"CONTEXTO:\n{_build_context(relevantes)}\n\nPREGUNTA: {question}"},
    ]
    respuesta = chat(messages)

    # El LLM es el juez final de relevancia (grounding): puede concluir "no encontrado"
    # aunque la búsqueda haya traído fragmentos.
    encontrado = "no encontré esa información" not in respuesta.lower()
    fuentes = [
        {"source": h.payload["source"], "section": h.payload.get("section"), "score": round(h.score, 3)}
        for h in relevantes
    ]
    return {"encontrado": encontrado, "respuesta": respuesta, "fuentes": fuentes if encontrado else []}
