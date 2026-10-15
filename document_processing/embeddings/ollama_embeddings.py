"""Ollama HTTP embedding provider adapter for local embeddings.

Connects to Ollama's official `/api/embed` endpoint, supporting local models
such as `embeddinggemma-2:270m` with asymmetric task formatting (query vs document),
finite numeric validation, batching, and structured error reporting.
"""

import math
import os
import logging
from typing import Any, Dict, List, Optional, Union

import requests

from document_processing.embeddings.base import EmbeddingProvider
from document_processing.models.chunk import Chunk

logger = logging.getLogger(__name__)


class OllamaEmbeddingError(RuntimeError):
    """Base exception for Ollama embedding errors."""
    pass


class OllamaConnectionError(OllamaEmbeddingError):
    """Raised when connection to the Ollama server fails."""
    pass


class OllamaTimeoutError(OllamaEmbeddingError):
    """Raised when an Ollama API request times out."""
    pass


class OllamaModelNotFoundError(OllamaEmbeddingError):
    """Raised when the specified model is not installed or found on the Ollama server."""
    pass


class OllamaAPIError(OllamaEmbeddingError):
    """Raised when Ollama returns an HTTP 4xx or 5xx error."""
    pass


class OllamaResponseError(OllamaEmbeddingError):
    """Raised when Ollama returns an unparseable or invalid response body."""
    pass


class OllamaEmbeddingProvider(EmbeddingProvider):
    """
    Local embedding provider communicating with Ollama's `POST /api/embed` API.
    Designed for `embeddinggemma-2:270m` and compatible local embedding models.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        batch_size: int = 32,
        timeout: float = 60.0,
        dimensions: Optional[int] = None,
    ):
        raw_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.base_url = raw_url.rstrip("/")
        self.model_name = model_name or os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma-2:270m")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Output dimensionality used to validate index and vector store compatibility."""
        if self._dimensions is None:
            self._dimensions = self._resolve_dimension()
        return self._dimensions

    @property
    def model_id(self) -> str:
        """Stable model identifier recorded alongside chunk vectors."""
        return f"ollama/{self.model_name}:{self.dimensions}d"

    @property
    def normalized(self) -> bool:
        """EmbeddingGemma 2 outputs are unit-normalized (L2 norm = 1.0)."""
        return True

    def _resolve_dimension(self) -> int:
        """
        Resolves model dimensionality from environment hints or live probe.
        Defaults to 768 for embeddinggemma-2.
        """
        env_dim = (
            os.getenv("EMBEDDING_DIMENSIONS")
            or os.getenv("OLLAMA_EMBEDDING_DIMENSIONS")
            or os.getenv("LOCAL_EMBEDDING_DIMENSIONS")
        )
        if env_dim:
            try:
                return int(env_dim)
            except ValueError:
                pass

        try:
            sample = self._call_api(["probe text"])
            if sample and len(sample[0]) > 0:
                return len(sample[0])
        except Exception as e:
            logger.debug(f"Could not probe Ollama dimension dynamically: {e}")

        return 768

    @staticmethod
    def preprocess_query(query: str) -> str:
        """
        Formats search query for EmbeddingGemma 2.
        Format: 'task: search result | query: {query}'
        Avoids duplicate prefixing if query already begins with 'task:'.
        """
        stripped = query.strip()
        if not stripped:
            raise ValueError("Query text cannot be empty.")
        if stripped.lower().startswith("task:"):
            return stripped
        return f"task: search result | query: {stripped}"

    @staticmethod
    def preprocess_document(content: str, title: Optional[str] = None) -> str:
        """
        Formats document chunk content for EmbeddingGemma 2.
        Format: 'title: {title} | text: {content}' or 'title: none | text: {content}'
        Avoids duplicate prefixing if content already begins with 'title:'.
        """
        stripped_content = content.strip()
        if not stripped_content:
            raise ValueError("Document chunk content cannot be empty.")
        if stripped_content.lower().startswith("title:"):
            return stripped_content

        doc_title = title.strip() if title and title.strip() else "none"
        return f"title: {doc_title} | text: {stripped_content}"

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """
        Calls Ollama's POST /api/embed endpoint with a batch of texts.
        Validates response structure, vector count, dimension, and finiteness.
        """
        if not texts:
            return []

        url = f"{self.base_url}/api/embed"
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
        }

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Failed to connect to Ollama at '{self.base_url}'. "
                "Ensure the Ollama service is running (e.g., 'ollama serve')."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(
                f"Request to Ollama at '{self.base_url}' timed out after {self.timeout}s."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaEmbeddingError(f"HTTP request to Ollama failed: {exc}") from exc

        if response.status_code == 404:
            try:
                err_data = response.json()
                msg = err_data.get("error", response.text)
            except Exception:
                msg = response.text
            raise OllamaModelNotFoundError(
                f"Ollama model '{self.model_name}' was not found on server at '{self.base_url}'. "
                f"Run 'ollama pull {self.model_name}' to install it. Details: {msg}"
            )

        if response.status_code != 200:
            try:
                err_data = response.json()
                msg = err_data.get("error", response.text)
            except Exception:
                msg = response.text
            raise OllamaAPIError(
                f"Ollama embedding request failed (HTTP {response.status_code}): {msg}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError(f"Ollama response could not be parsed as JSON: {response.text[:200]}") from exc

        if not isinstance(data, dict) or "embeddings" not in data:
            raise OllamaResponseError(
                f"Malformed Ollama response: expected dict with 'embeddings' key, got: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )

        embeddings = data["embeddings"]
        if not isinstance(embeddings, list):
            raise OllamaResponseError(f"Expected 'embeddings' to be a list, got {type(embeddings)}")

        if len(embeddings) != len(texts):
            raise OllamaResponseError(
                f"Input count mismatch: sent {len(texts)} texts, received {len(embeddings)} embeddings."
            )

        # Validate vector dimensions and numeric finiteness
        expected_dim = self._dimensions
        for idx, vec in enumerate(embeddings):
            if not isinstance(vec, list) or len(vec) == 0:
                raise OllamaResponseError(f"Embedding at index {idx} is empty or not a list.")

            if expected_dim is None:
                expected_dim = len(vec)
                self._dimensions = expected_dim
            elif len(vec) != expected_dim:
                raise OllamaResponseError(
                    f"Dimension mismatch at index {idx}: expected {expected_dim}, got {len(vec)}."
                )

            for val_idx, val in enumerate(vec):
                if not isinstance(val, (int, float)) or not math.isfinite(val):
                    raise OllamaResponseError(
                        f"Non-finite or non-numeric value ({val}) at vector {idx}, position {val_idx}."
                    )

        return embeddings

    def embed_texts(
        self,
        texts: Union[str, List[str]],
        *,
        is_query: bool = False,
        titles: Optional[List[Optional[str]]] = None,
    ) -> List[List[float]]:
        """
        Generates embeddings for one or more strings in batches.
        Preserves input order and returns one vector per input text.
        """
        if isinstance(texts, str):
            input_list = [texts]
        else:
            input_list = list(texts)

        if not input_list:
            return []

        # Preprocess query vs document
        if is_query:
            formatted = [self.preprocess_query(t) for t in input_list]
        else:
            if titles and len(titles) == len(input_list):
                formatted = [
                    self.preprocess_document(t, title=titles[i])
                    for i, t in enumerate(input_list)
                ]
            else:
                formatted = [self.preprocess_document(t) for t in input_list]

        all_vectors: List[List[float]] = []
        for i in range(0, len(formatted), self.batch_size):
            batch = formatted[i : i + self.batch_size]
            vectors = self._call_api(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a single query using query-specific instruction formatting.
        """
        vectors = self.embed_texts(query, is_query=True)
        if not vectors:
            raise OllamaResponseError("No vector returned for search query.")
        return vectors[0]

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generates embeddings for a list of document chunks in batches.
        Attaches the resulting vector to each chunk's `embedding` attribute.
        """
        pending = [c for c in chunks if c.embedding is None]
        if not pending:
            return chunks

        titles = [c.metadata.document_title if c.metadata else None for c in pending]
        contents = [c.content for c in pending]

        logger.debug(f"Generating Ollama embeddings for {len(pending)} chunks (batch_size={self.batch_size})")
        vectors = self.embed_texts(contents, is_query=False, titles=titles)

        if len(vectors) != len(pending):
            raise OllamaResponseError(
                f"Expected {len(pending)} vectors for chunks, received {len(vectors)}."
            )

        for chunk, vector in zip(pending, vectors):
            chunk.embedding = vector

        return chunks
