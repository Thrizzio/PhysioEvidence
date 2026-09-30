import logging
from typing import List, Optional, Union, Callable
from storage.base import VectorStore, SearchResult, FilterCriteria
from document_processing.embeddings.base import EmbeddingProvider
from retrieval.base import BaseRetriever
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

class VectorRetriever(BaseRetriever):
    """
    Standard dense semantic retriever.
    Converts input query into an embedding vector and retrieves top-k nearest chunks
    from the underlying VectorStore using cosine similarity.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Union[EmbeddingProvider, Callable[[str], List[float]]]
    ):
        self.vector_store = vector_store
        self.embedder = embedder

    def _get_query_embedding(self, query: str) -> List[float]:
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(query)
        elif callable(self.embedder):
            return self.embedder(query)
        else:
            raise TypeError("Embedder must implement embed_query or be a callable taking a string.")

    @trace_operation(name="vector_retrieval", span_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        logger.debug(f"Executing vector retrieval for query: '{query}' (top_k={top_k})")
        query_embedding = self._get_query_embedding(query.strip())
        
        results = self.vector_store.search_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_criteria=filter_criteria
        )
        logger.debug(f"Retrieved {len(results)} chunks via vector similarity")
        return results
