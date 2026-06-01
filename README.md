# Asistente de Soporte — MineCatalog

Asistente automatizado que responde preguntas de soporte usando **únicamente** la
documentación interna del software *MineCatalog*. Si la información no está en la
documentación, lo indica explícitamente (no inventa).

**Stack:** n8n (Webhook HTTP) · Python + FastAPI (motor de recuperación y generación) ·
Qdrant (búsqueda semántica) · LLM configurable (OpenAI / Ollama).

## Arquitectura

```
Usuario
  │  pregunta (HTTP POST)
  ▼
[ n8n ]  Webhook HTTP ──→ validación ──→ orquestación
  │  HTTP Request
  ▼
[ API Python (FastAPI) ]  POST /query
  ├─ 1. Embedding de la pregunta
  ├─ 2. Búsqueda semántica en Qdrant (top-k chunks)
  ├─ 3. Armado de contexto + prompt ("no inventar")
  └─ 4. LLM → respuesta con citas de fuente
  ▼
Respuesta JSON { encontrado, respuesta, fuentes }
```

**Ingesta (offline):** `docs/` (.txt, .md, .pdf, .json) → limpieza → normalización →
chunking por sección → embeddings → índice Qdrant.

---

## Opción A — Levantar con Docker (recomendado)

Levanta **API + n8n + Qdrant** con un solo comando. No requiere instalar Python,
Node ni Qdrant a mano.

### Requisitos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- Un proveedor de LLM:
  - **OpenAI** — una API key con saldo
  - **Ollama** instalado en el host (local, gratis)

### Pasos

```bash
# 1. Configurar el proveedor de LLM
cp .env.example .env
# Editar .env: poner OPENAI_API_KEY (o configurar Ollama)

# 2. Levantar los servicios
docker compose up -d --build

# 3. Generar el índice (una sola vez)
docker compose run --rm -e QDRANT_URL=http://qdrant:6333 api python -m src.ingest

# 4. Importar el workflow en n8n
#    Abrir http://localhost:5678 en el navegador
#    Crear una cuenta (es local, cualquier dato)
#    Menú lateral → Workflows → Import from File → seleccionar n8n/workflow.docker.json
#    Activar el workflow (toggle arriba a la derecha)
```

### Probar

```bash
# Pregunta con respuesta:
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "no se conecta a la base de datos"}'

# Pregunta sin respuesta (debe decir que no encontró):
curl -X POST http://localhost:5678/webhook/soporte \
  -H "Content-Type: application/json" \
  -d '{"pregunta": "que significa el error 502"}'
```

En PowerShell:
```powershell
Invoke-RestMethod -Uri http://localhost:5678/webhook/soporte -Method Post -ContentType "application/json" -Body '{"pregunta":"no se conecta a la base de datos"}' | Format-List
```

> **Nota Ollama + Docker:** si los embeddings usan Ollama del host, poner
> `OLLAMA_HOST=http://host.docker.internal:11434` en `.env` y asegurar que
> Ollama esté corriendo.

### Detener

```bash
docker compose down
```

---

## Opción B — Levantar sin Docker

### Requisitos

- Python 3.12+
- Node.js (para n8n)
- Un proveedor de LLM:
  - **Ollama** instalado y corriendo (`ollama serve`) — local, gratis
  - **OpenAI** — una API key con saldo

### Pasos

```powershell
# 1. Instalar dependencias
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configurar
Copy-Item .env.example .env
# Editar .env: elegir proveedor y poner la API key correspondiente

# 3. (Solo con Ollama) Descargar modelos
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 4. Generar el índice
python -m src.ingest

# 5. Levantar la API (Terminal 1)
uvicorn src.api:app --host 0.0.0.0 --port 8000

# 6. Levantar n8n (Terminal 2 — abrir otra ventana de PowerShell)
npx n8n
```

Luego en n8n (http://localhost:5678):
1. Crear cuenta (local, cualquier dato).
2. Menú → Workflows → **Import from File** → seleccionar `n8n/workflow.json`.
3. **Activar** el workflow (toggle arriba a la derecha).

### Probar

```powershell
# Webhook (n8n → API → LLM):
Invoke-RestMethod -Uri http://localhost:5678/webhook/soporte -Method Post -ContentType "application/json" -Body '{"pregunta":"me dice usuario o contrasena incorrectos"}' | Format-List

# API directo:
Invoke-RestMethod -Uri http://localhost:8000/query -Method Post -ContentType "application/json" -Body '{"pregunta":"el catalogo carga lento"}' | Format-List

# Chat interactivo de consola:
python chat.py

# Documentación interactiva en el navegador:
# http://localhost:8000/docs → POST /query → Try it out
```

---

## Preguntas de prueba

| Pregunta | Resultado esperado |
|---|---|
| `no se conecta a la base de datos` | Responde con causas y solución, cita fuentes |
| `me dice usuario o contraseña incorrectos` | Responde con ERR-AUTH-001 |
| `el catalogo carga lento` | Responde con causas y acciones recomendadas |
| `codigo de material duplicado` | Responde citando ambas fuentes (.txt y .json) |
| `me sale permiso denegado` | Responde con causas y solución |
| `como contacto al soporte tecnico` | Da correo, horario y niveles de soporte |
| `que significa el error 502` | "No encontré esa información…" (correcto: no está en la doc) |
| `no puedo acceder al dashboard` | "No encontré esa información…" (correcto: no está en la doc) |
| *(vacío)* | "La consulta está vacía…" |

## Respuesta de la API

```json
{
  "encontrado": true,
  "respuesta": "El error indica que... [Documentación 4.json · ERR-DB-001].",
  "fuentes": [
    {"source": "Documentación 2.txt", "section": "3.2", "score": 0.784},
    {"source": "Documentación 4.json", "section": "ERR-DB-001", "score": 0.739}
  ]
}
```

## Estructura del proyecto

```
src/
  config.py             # configuración desde .env
  ingest.py             # ingesta: parseo + limpieza + chunking + indexado
  embeddings.py         # embeddings (OpenAI / Ollama)
  vectorstore.py        # Qdrant: indexación + búsqueda semántica
  llm.py                # cliente de chat (OpenAI / Ollama)
  rag.py                # recuperación + prompt de grounding + generación
  api.py                # FastAPI: POST /query, GET /health
docs/                   # corpus a ingerir (.txt, .md, .pdf, .json)
n8n/
  workflow.json         # workflow para uso local (URL 127.0.0.1)
  workflow.docker.json  # workflow para Docker (URL http://api:8000)
chat.py                 # chat interactivo de consola
Dockerfile              # imagen de la API
docker-compose.yml      # orquesta: qdrant + api + n8n
.env.example            # variables de entorno de ejemplo
.gitignore
requirements.txt
```
