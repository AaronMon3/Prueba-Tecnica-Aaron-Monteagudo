"""Almacén vectorial sobre Qdrant.

Dos modos según configuración:
- Embebido (por defecto): persistido en disco local, sin servidor ni Docker.
- Servidor: si QDRANT_URL está definido (p. ej. en Docker), se conecta a ese Qdrant.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .config import settings


class VectorStore:
    def __init__(self, path: str | None = None, collection: str | None = None) -> None:
        self.collection = collection or settings.qdrant_collection
        if settings.qdrant_url:  # Qdrant como servicio (Docker)
            self.client = QdrantClient(url=settings.qdrant_url)
        else:  # Qdrant embebido en disco (sin servidor ni Docker)
            self.client = QdrantClient(path=path or settings.qdrant_path)

    def reset(self, dim: int) -> None:
        """(Re)crea la colección vacía con la dimensión de vector indicada."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    def add(self, vectors: list[list[float]], payloads: list[dict]) -> int:
        """Inserta los vectores con su metadata. Devuelve la cantidad insertada."""
        points = [
            PointStruct(id=i, vector=vec, payload=payload)
            for i, (vec, payload) in enumerate(zip(vectors, payloads))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)

    def search(self, vector: list[float], top_k: int):
        """Búsqueda semántica. Devuelve una lista de puntos con `.score` y `.payload`."""
        result = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return result.points

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
