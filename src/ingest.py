"""Pipeline de ingesta.

Lee `docs/` (.txt, .md, .pdf, .json), normaliza el contenido y aplica
chunking por sección. Cada entrada se enriquece con metadata (fuente, sección,
título, palabras clave) que se antepone antes de generar el embedding.
Finalmente indexa todo en Qdrant.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

from .config import settings

MAX_TOKENS = 512  # tope por chunk; las entradas suelen ser mucho más cortas
SUPPORTED = {".txt", ".pdf", ".md", ".json"}
_enc = tiktoken.get_encoding("cl100k_base")

# Sub-etiquetas internas de una entrada (no son títulos de sección).
SUBLABELS = {
    "posibles causas", "causas posibles", "acciones recomendadas",
    "acción recomendada", "accion recomendada", "verificaciones básicas",
    "verificaciones basicas", "mensaje mostrado", "solución", "solucion",
    "información recomendada al reportar un incidente",
}


@dataclass
class Entry:
    """Una unidad de sentido del documento (un error/problema)."""
    source: str
    section: str
    title: str
    body: str
    keywords: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)


@dataclass
class Chunk:
    embed_text: str   # texto que se embebe (con contexto antepuesto)
    payload: dict     # metadata + texto legible para el contexto del LLM


# --------------------------------------------------------------------------- #
# Utilidades de texto
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Normaliza unicode (ligaduras 'ﬁ'->'fi'), viñetas, espacios y control."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    text = re.sub(r"[•·▪◦∙][ \t]*\n?[ \t]*", "- ", text)  # viñeta suelta -> guion
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def split_by_tokens(text: str, max_tokens: int = MAX_TOKENS - 32) -> list[str]:
    """Fallback recursivo: divide un texto largo respetando párrafos y oraciones."""
    paragraphs = [p for p in re.split(r"\n\n+", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip()
        if count_tokens(candidate) <= max_tokens:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if count_tokens(para) <= max_tokens:
                current = para
            else:  # párrafo muy largo: dividir por oraciones
                current = ""
                for sentence in re.split(r"(?<=[.!?])\s+", para):
                    cand = f"{current} {sentence}".strip()
                    if count_tokens(cand) <= max_tokens:
                        current = cand
                    else:
                        if current:
                            chunks.append(current)
                        current = sentence
    if current:
        chunks.append(current)
    return chunks or [text]


# --------------------------------------------------------------------------- #
# Parsers por formato
# --------------------------------------------------------------------------- #
_HEAD_RE = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)$")


def parse_txt(path: Path) -> list[Entry]:
    """Texto plano con encabezados 'N.N Título' (ej. Documentación 2.txt)."""
    entries: list[Entry] = []
    title: str | None = None
    section, body = "", []
    for line in path.read_text(encoding="utf-8").split("\n"):
        match = _HEAD_RE.match(line.strip())
        if match:
            if title is not None:
                entries.append(Entry(path.name, section, title, clean_text("\n".join(body))))
            section, title, body = match.group(1), match.group(2), []
        elif title is not None:
            body.append(line)
    if title is not None:
        entries.append(Entry(path.name, section, title, clean_text("\n".join(body))))
    return entries


def parse_pdf(path: Path) -> list[Entry]:
    """PDF: extrae el texto por líneas y separa por encabezados de problema.

    En este corpus el tamaño de fuente no distingue títulos (todo a 12pt), así que
    un encabezado es: una línea numerada 'N.N ...' o una línea en negrita que no sea
    una sub-etiqueta interna (Posibles causas, etc.) ni contenido (con '>' o ':').
    """
    import fitz  # PyMuPDF

    lines: list[tuple[str, bool]] = []
    doc = fitz.open(path)
    for page in doc:
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text = unicodedata.normalize("NFKC", "".join(s.get("text", "") for s in spans)).strip()
                if not text:
                    continue
                bold = any((s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower() for s in spans)
                lines.append((text, bold))
    doc.close()

    def is_heading(text: str, bold: bool) -> bool:
        if _HEAD_RE.match(text):
            return True
        if not bold:
            return False
        norm = text.rstrip(":").strip().lower()
        if norm in SUBLABELS or ">" in text or text.endswith(":") or len(text) > 60:
            return False
        return True

    entries: list[Entry] = []
    title: str | None = None
    section, body = "", []
    for text, bold in lines:
        if is_heading(text, bold):
            if title is not None:
                entries.append(Entry(path.name, section, title, clean_text("\n".join(body))))
            match = _HEAD_RE.match(text)
            section, title = (match.group(1), match.group(2)) if match else ("", text)
            body = []
        elif title is not None:
            body.append(text)
    if title is not None:
        entries.append(Entry(path.name, section, title, clean_text("\n".join(body))))
    return entries


def parse_md(path: Path) -> list[Entry]:
    """Markdown: cada encabezado H1 (#) es una entrada; extrae campos de los H2 (##)."""
    text = path.read_text(encoding="utf-8")
    entries: list[Entry] = []
    for part in re.split(r"(?m)^#\s+", text):
        part = part.strip()
        if not part:
            continue
        head, _, rest = part.partition("\n")
        title = head.strip()
        sections = {
            m.group(1).strip().lower(): m.group(2).strip()
            for m in re.finditer(r"(?m)^##\s+(.+?)\s*\n(.*?)(?=^##\s+|\Z)", rest, re.S)
        }
        keywords: list[str] = []
        if "palabras clave" in sections:
            keywords = [k.strip() for k in re.split(r"[,\n]", sections["palabras clave"]) if k.strip()]
        error_code = sections.get("código") or sections.get("codigo")
        body = clean_text(re.sub(r"(?m)^##\s+", "", rest))
        entries.append(Entry(path.name, error_code or "", title, body, keywords, {"error_code": error_code}))
    return entries


def parse_json(path: Path) -> list[Entry]:
    """JSON estructurado: cada objeto de 'contenido' es una entrada."""
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("contenido") if isinstance(data, dict) else None
    if not isinstance(items, list):
        body = clean_text(json.dumps(data, ensure_ascii=False, indent=2))
        return [Entry(path.name, "", path.stem, body)]

    entries: list[Entry] = []
    for obj in items:
        parts: list[str] = []
        if obj.get("mensaje_usuario"):
            parts.append(f"Mensaje al usuario: {obj['mensaje_usuario']}")
        if obj.get("causas_posibles"):
            parts.append("Causas posibles:\n" + "\n".join(f"- {c}" for c in obj["causas_posibles"]))
        if obj.get("solucion"):
            parts.append("Solución:\n" + "\n".join(f"- {s}" for s in obj["solucion"]))
        if obj.get("nivel_soporte"):
            parts.append(f"Nivel de soporte: {obj['nivel_soporte']}")
        entries.append(Entry(
            source=path.name,
            section=obj.get("id", ""),
            title=obj.get("titulo", ""),
            body=clean_text("\n".join(parts)),
            keywords=obj.get("palabras_clave", []) or [],
            extra={"error_code": obj.get("id"), "nivel_soporte": obj.get("nivel_soporte")},
        ))
    return entries


_PARSERS = {".txt": parse_txt, ".pdf": parse_pdf, ".md": parse_md, ".json": parse_json}


def parse_file(path: Path) -> list[Entry]:
    parser = _PARSERS.get(path.suffix.lower())
    return parser(path) if parser else []


# --------------------------------------------------------------------------- #
# Entrada -> chunks (metadata-aware)
# --------------------------------------------------------------------------- #
def to_chunks(entry: Entry) -> list[Chunk]:
    location = " ".join(x for x in (entry.section, entry.title) if x)
    header = f"[Fuente: {entry.source} | Sección: {location}]"
    kw = f"\nPalabras clave: {', '.join(entry.keywords)}" if entry.keywords else ""
    full = f"{header}\n{entry.body}{kw}"
    bodies = [entry.body] if count_tokens(full) <= MAX_TOKENS else split_by_tokens(entry.body)

    chunks: list[Chunk] = []
    for i, body in enumerate(bodies):
        embed_text = f"{header}\n{body}{kw}"
        readable = f"{entry.title}\n{body}".strip()
        chunks.append(Chunk(
            embed_text=embed_text,
            payload={
                "source": entry.source,
                "section": entry.section,
                "title": entry.title,
                "keywords": entry.keywords,
                "error_code": entry.extra.get("error_code"),
                "nivel_soporte": entry.extra.get("nivel_soporte"),
                "text": readable,
                "part": i,
            },
        ))
    return chunks


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # Evita errores de codificación al imprimir en consolas Windows (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Ingesta del corpus a Qdrant.")
    ap.add_argument("--dry-run", action="store_true", help="Solo parsea y muestra los chunks.")
    ap.add_argument("--docs", default=None, help="Carpeta del corpus (por defecto ./docs).")
    args = ap.parse_args()

    docs_dir = Path(args.docs) if args.docs else Path(__file__).resolve().parent.parent / "docs"
    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED)

    all_chunks: list[Chunk] = []
    print(f"Leyendo {len(files)} archivo(s) de {docs_dir}\n")
    for f in files:
        entries = parse_file(f)
        file_chunks = [c for e in entries for c in to_chunks(e)]
        all_chunks.extend(file_chunks)
        print(f"  {f.name}: {len(entries)} entradas -> {len(file_chunks)} chunks")
    print(f"\nTotal: {len(all_chunks)} chunks")

    if args.dry_run:
        print("\n--- DRY RUN: vista previa ---")
        for c in all_chunks:
            p = c.payload
            preview = p["text"][:180].replace("\n", " ")
            print(f"\n[{p['source']} | {p['section']} {p['title']}]  kw={p['keywords']}")
            print(f"    {preview}{'...' if len(p['text']) > 180 else ''}")
        return

    from .embeddings import embed_texts
    from .vectorstore import VectorStore

    print(f"\nGenerando embeddings (proveedor '{settings.embeddings_provider}', modelo '{settings.embedding_model}')...")
    vectors = embed_texts([c.embed_text for c in all_chunks])
    store = VectorStore()
    store.reset(dim=len(vectors[0]))
    n = store.add(vectors, [c.payload for c in all_chunks])
    store.close()
    destino = settings.qdrant_url or settings.qdrant_path
    print(f"Indexados {n} chunks en Qdrant (colección '{settings.qdrant_collection}', destino '{destino}').")


if __name__ == "__main__":
    main()
