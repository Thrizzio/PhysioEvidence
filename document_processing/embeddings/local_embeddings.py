"""Optional laptop embedding provider backed by EmbeddingGemma.

The heavyweight dependency is imported lazily so the core package and tests do
not download model weights. Install `requirements-local-embeddings.txt` before
selecting this provider.
"""

from typing import List

from document_processing.embeddings.base import EmbeddingProvider
from document_processing.models.chunk import Chunk


class LocalEmbeddingProvider(EmbeddingProvider):
    """EmbeddingGemma provider using sentence-transformers encode APIs."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-300m",
        batch_size: int = 16,
        dimensions: int = 768,
    ):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local embeddings require sentence-transformers; install "
                "requirements-local-embeddings.txt first."
            ) from exc
        if dimensions not in (128, 256, 512, 768):
            raise ValueError("EmbeddingGemma dimensions must be one of 128, 256, 512, or 768.")
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimensions = dimensions
        self._model = SentenceTransformer(model_name)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_id(self) -> str:
        return f"{self.model_name}:{self._dimensions}d"

    @property
    def normalized(self) -> bool:
        return True

    def _encode(self, texts: List[str], *, query: bool) -> List[List[float]]:
        method_name = "encode_query" if query and hasattr(self._model, "encode_query") else (
            "encode_document" if not query and hasattr(self._model, "encode_document") else "encode"
        )
        method = getattr(self._model, method_name)
        encoded = method(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vectors = encoded.tolist()
        if self._dimensions != len(vectors[0]) if vectors else False:
            vectors = [vector[: self._dimensions] for vector in vectors]
        return vectors

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        pending = [chunk for chunk in chunks if chunk.embedding is None]
        if not pending:
            return chunks
        vectors = self._encode([chunk.content for chunk in pending], query=False)
        if len(vectors) != len(pending) or any(len(vector) != self.dimensions for vector in vectors):
            raise RuntimeError("Local embedding provider returned an incompatible vector shape.")
        for chunk, vector in zip(pending, vectors):
            chunk.embedding = vector
        return chunks

    def embed_query(self, query: str) -> List[float]:
        vectors = self._encode([query], query=True)
        if not vectors or len(vectors[0]) != self.dimensions:
            raise RuntimeError("Local embedding provider returned an incompatible query vector shape.")
        return vectors[0]
