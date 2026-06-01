"""Configuración central del proyecto, leída desde variables de entorno (.env)."""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    def __init__(self) -> None:
        # Proveedor de IA: "openai" (o endpoint compatible) u "ollama".
        # Se puede separar el proveedor de chat y el de embeddings (p. ej. chat por
        # API remota + embeddings locales). Si no se especifican, usan LLM_PROVIDER.
        self.provider = os.getenv("LLM_PROVIDER", "ollama").lower()
        self.chat_provider = os.getenv("CHAT_PROVIDER", self.provider).lower()
        self.embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", self.provider).lower()

        # OpenAI (o cualquier endpoint compatible vía OPENAI_BASE_URL)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.openai_embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()

        # Ollama
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
        self.ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

        # RAG
        self.top_k = int(os.getenv("TOP_K", "4"))
        self.score_threshold = float(os.getenv("SCORE_THRESHOLD", "0.35"))
        self.qdrant_path = os.getenv("QDRANT_PATH", "./qdrant_data")
        self.qdrant_url = os.getenv("QDRANT_URL", "").strip()  # si está, usa Qdrant servidor (Docker)
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "minecatalog")

        # API
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.request_timeout = float(os.getenv("REQUEST_TIMEOUT", "60"))

    @property
    def chat_model(self) -> str:
        return self.openai_chat_model if self.chat_provider == "openai" else self.ollama_chat_model

    @property
    def embedding_model(self) -> str:
        return self.openai_embedding_model if self.embeddings_provider == "openai" else self.ollama_embedding_model


settings = Settings()
