import logging
from typing import List, Optional, Union, Callable, Dict
from storage.base import VectorStore, SearchResult, FilterCriteria
from document_processing.embeddings.base import EmbeddingProvider
from retrieval.base import BaseRetriever
from retrieval.vector_retriever import VectorRetriever
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    dense_results: List[SearchResult],
    sparse_results: List[SearchResult],
    k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    top_k: int = 5
) -> List[SearchResult]:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum( weight / (k + rank) ) across search channels.
    """
    scores: Dict[str, float] = {}
    chunk_map: Dict[str, SearchResult] = {}

    # Accumulate dense channel rank scores (1-based rank)
    for rank, item in enumerate(dense_results, start=1):
        chunk_map[item.chunk_id] = item
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + (dense_weight / (k + rank))

    # Accumulate sparse channel rank scores (1-based rank)
    for rank, item in enumerate(sparse_results, start=1):
        if item.chunk_id not in chunk_map:
            chunk_map[item.chunk_id] = item
        scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + (sparse_weight / (k + rank))

    # Sort candidates by fused score descending
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    fused_results: List[SearchResult] = []
    for chunk_id, fused_score in sorted_items[:top_k]:
        original = chunk_map[chunk_id]
        # Attach the RRF score while preserving the original chunk data and metadata
        fused_item = original.model_copy(update={"score": round(fused_score, 6)})
        fused_results.append(fused_item)

    return fused_results

class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining dense semantic search (pgvector) and sparse lexical search (full-text)
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Union[EmbeddingProvider, Callable[[str], List[float]]],
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_multiplier: int = 3
    ):
        self.vector_store = vector_store
        self.vector_retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_multiplier = candidate_multiplier

    @trace_operation(name="hybrid_retrieval", span_type="retriever")
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        candidate_pool_size = max(top_k * self.candidate_multiplier, 10)
        clean_query = query.strip()

        # 1. Fetch dense candidates
        dense_candidates = self.vector_retriever.retrieve(
            query=clean_query,
            top_k=candidate_pool_size,
            filter_criteria=filter_criteria
        )

        # 2. Fetch sparse lexical candidates
        sparse_candidates = self.vector_store.search_text(
            query_text=clean_query,
            top_k=candidate_pool_size,
            filter_criteria=filter_criteria
        )

        # 3. Fuse via Reciprocal Rank Fusion
        fused = reciprocal_rank_fusion(
            dense_results=dense_candidates,
            sparse_results=sparse_candidates,
            k=self.rrf_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight,
            top_k=top_k
        )

        logger.debug(
            f"Hybrid retrieval: {len(dense_candidates)} dense, {len(sparse_candidates)} sparse "
            f"fused to {len(fused)} results."
        )
        return fused
