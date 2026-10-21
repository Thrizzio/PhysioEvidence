import math
import logging
from typing import List, Optional, Dict, Union, Any
from storage.base import VectorStore, SearchResult, FilterCriteria
from document_processing.models.document import ExtractedDocument, StructuredDocument
from document_processing.models.chunk import Chunk

logger = logging.getLogger(__name__)

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes cosine similarity between two float vectors.
    Returns 1.0 for identical vectors, 0.0 for orthogonal vectors.
    """
    if len(v1) != len(v2):
        raise ValueError(f"Vector dimensions do not match: {len(v1)} != {len(v2)}")
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot / (norm_v1 * norm_v2)

class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store implementing the VectorStore interface.
    Used for unit testing, offline development, and deterministic retrieval experiments.
    """

    def __init__(self):
        self.documents: Dict[str, Union[ExtractedDocument, StructuredDocument]] = {}
        self.chunks: Dict[str, Chunk] = {}

    def save_document(self, document: Union[ExtractedDocument, StructuredDocument]) -> None:
        self.documents[document.document_id] = document

    def save_chunks(self, chunks: List[Chunk]) -> None:
        for c in chunks:
            self.chunks[c.chunk_id] = c

    def _matches_filters(self, chunk: Chunk, filter_criteria: Optional[FilterCriteria]) -> bool:
        if not filter_criteria:
            return True

        if filter_criteria.document_ids and chunk.document_id not in filter_criteria.document_ids:
            return False

        if filter_criteria.sections:
            chunk_section = chunk.metadata.section if chunk.metadata else None
            if not chunk_section or chunk_section not in filter_criteria.sections:
                return False

        doc = self.documents.get(chunk.document_id)
        if filter_criteria.min_year is not None and (not doc or not doc.metadata or doc.metadata.publication_year is None):
            return False
        if filter_criteria.max_year is not None and (not doc or not doc.metadata or doc.metadata.publication_year is None):
            return False
        if filter_criteria.authors_contain and (not doc or not doc.metadata or not doc.metadata.authors):
            return False
        if doc and doc.metadata:
            meta = doc.metadata
            if filter_criteria.min_year is not None:
                if meta.publication_year is None or meta.publication_year < filter_criteria.min_year:
                    return False
            if filter_criteria.max_year is not None:
                if meta.publication_year is None or meta.publication_year > filter_criteria.max_year:
                    return False
            if filter_criteria.authors_contain:
                authors_str = " ".join(meta.authors or [])
                if filter_criteria.authors_contain.lower() not in authors_str.lower():
                    return False

        if filter_criteria.embedding_model:
            chunk_model = getattr(chunk, "embedding_model", None)
            if not chunk_model and chunk.metadata and chunk.metadata.extra:
                chunk_model = chunk.metadata.extra.get("embedding_model")
            if chunk_model and chunk_model != filter_criteria.embedding_model:
                return False

        return True

    def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        candidates: List[SearchResult] = []

        for chunk in self.chunks.values():
            if chunk.embedding is None:
                continue
            if not self._matches_filters(chunk, filter_criteria):
                continue

            sim = cosine_similarity(query_embedding, chunk.embedding)
            doc = self.documents.get(chunk.document_id)
            doc_title = doc.metadata.title if (doc and doc.metadata) else (chunk.metadata.document_title if chunk.metadata else None)

            candidates.append(SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section=chunk.metadata.section if chunk.metadata else None,
                page_start=chunk.metadata.page_start if chunk.metadata else None,
                page_end=chunk.metadata.page_end if chunk.metadata else None,
                score=sim,
                document_title=doc_title,
                authors=doc.metadata.authors if doc else None,
                publication_year=doc.metadata.publication_year if doc else None,
                journal=doc.metadata.journal if doc else None,
                doi=doc.metadata.doi if doc else None,
                source_url=doc.metadata.source_url if doc else None,
                metadata=chunk.metadata.extra if chunk.metadata else {}
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """
        Simple keyword overlap ranking for in-memory lexical matching.
        """
        query_words = set(query_text.lower().split())
        if not query_words:
            return []

        candidates: List[SearchResult] = []

        for chunk in self.chunks.values():
            if not self._matches_filters(chunk, filter_criteria):
                continue

            content_lower = chunk.content.lower()
            # Simple TF-based score
            matched_count = sum(content_lower.count(w) for w in query_words)
            if matched_count == 0:
                continue

            score = matched_count / max(1, len(chunk.content.split()))

            doc = self.documents.get(chunk.document_id)
            doc_title = doc.metadata.title if (doc and doc.metadata) else (chunk.metadata.document_title if chunk.metadata else None)

            candidates.append(SearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                section=chunk.metadata.section if chunk.metadata else None,
                page_start=chunk.metadata.page_start if chunk.metadata else None,
                page_end=chunk.metadata.page_end if chunk.metadata else None,
                score=score,
                document_title=doc_title,
                authors=doc.metadata.authors if doc else None,
                publication_year=doc.metadata.publication_year if doc else None,
                journal=doc.metadata.journal if doc else None,
                doi=doc.metadata.doi if doc else None,
                source_url=doc.metadata.source_url if doc else None,
                metadata=chunk.metadata.extra if chunk.metadata else {}
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:top_k]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def get_chunks_for_document(self, document_id: str) -> List[Chunk]:
        matching = [c for c in self.chunks.values() if c.document_id == document_id]
        matching.sort(key=lambda c: c.chunk_index)
        return matching

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self.documents:
            return False
        del self.documents[document_id]
        self.chunks = {k: v for k, v in self.chunks.items() if v.document_id != document_id}
        return True

    def count_documents(self) -> int:
        return len(self.documents)

    def has_document(self, document_id: str) -> bool:
        """
        Checks if a document exists by document_id or content_hash.
        """
        if document_id in self.documents:
            return True
        for doc in self.documents.values():
            if getattr(doc, "document_id", None) == document_id:
                return True
            if hasattr(doc, "metadata") and doc.metadata:
                if getattr(doc.metadata, "content_hash", None) == document_id:
                    return True
        return False

    def count_chunks(self) -> int:
        return len(self.chunks)

