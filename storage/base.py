from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from document_processing.models.document import ExtractedDocument, StructuredDocument
from document_processing.models.chunk import Chunk

class FilterCriteria(BaseModel):
    """
    Filtering criteria for vector and text search queries.
    """
    document_ids: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    min_year: Optional[int] = None
    max_year: Optional[int] = None
    authors_contain: Optional[str] = None
    embedding_model: Optional[str] = None
    extra_filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    """
    Represents a retrieved chunk with similarity score and contextual metadata.
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    score: float  # Cosine similarity (1.0 = identical) or text rank score
    document_title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    source_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorStore(ABC):
    """
    Abstract interface for document and vector storage.
    Enables swapping PostgreSQL + pgvector with in-memory stores for testing.
    """

    @abstractmethod
    def save_document(self, document: Union[ExtractedDocument, StructuredDocument]) -> None:
        """Saves or updates document metadata."""
        pass

    @abstractmethod
    def save_chunks(self, chunks: List[Chunk]) -> None:
        """Saves chunks and their vector embeddings."""
        pass

    @abstractmethod
    def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """Performs semantic similarity search using vector embeddings."""
        pass

    @abstractmethod
    def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """Performs lexical full-text search."""
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Retrieves a single chunk by ID."""
        pass

    @abstractmethod
    def get_chunks_for_document(self, document_id: str) -> List[Chunk]:
        """Retrieves all chunks associated with a document ordered by index."""
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Deletes a document and cascades to its chunks."""
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """Returns the total number of stored documents."""
        pass

    @abstractmethod
    def count_chunks(self) -> int:
        """Returns the total number of stored chunks."""
        pass
