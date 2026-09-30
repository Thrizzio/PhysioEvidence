from abc import ABC, abstractmethod
import logging
from typing import List, Optional, Callable
from storage.base import SearchResult
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

class BaseReranker(ABC):
    """
    Abstract interface for second-stage rerankers.
    Re-scores and re-orders a candidate set of retrieved chunks.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 5
    ) -> List[SearchResult]:
        """
        Reranks retrieved candidate chunks against the query and returns top_n.
        """
        pass

class CrossEncoderReranker(BaseReranker):
    """
    Reranker supporting custom scoring functions or cross-encoder models.
    Falls back gracefully to term-proximity and clinical section weighting if no neural model is attached.
    """

    def __init__(
        self,
        scoring_fn: Optional[Callable[[str, str], float]] = None,
        section_weights: Optional[dict] = None
    ):
        self.scoring_fn = scoring_fn
        # Scientific papers: empirical findings in 'Results' & 'Conclusions' carry high evidence weight
        self.section_weights = section_weights or {
            "results": 1.25,
            "conclusion": 1.25,
            "conclusions": 1.25,
            "discussion": 1.15,
            "abstract": 1.10,
            "methods": 1.05
        }

    @trace_operation(name="reranking", span_type="reranker")
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_n: int = 5
    ) -> List[SearchResult]:
        if not results:
            return []

        clean_query = query.lower().strip()
        query_words = set(clean_query.split())

        scored_results: List[SearchResult] = []

        for item in results:
            if self.scoring_fn is not None:
                # Use custom scoring function (e.g., neural cross-encoder)
                new_score = self.scoring_fn(query, item.content)
            else:
                # Heuristic cross-scoring: lexical coverage + section weighting
                content_lower = item.content.lower()
                
                # Term coverage ratio
                matched_words = sum(1 for w in query_words if w in content_lower)
                term_coverage = matched_words / max(1, len(query_words))
                
                # Exact phrase match boost
                phrase_boost = 1.3 if clean_query in content_lower else 1.0
                
                # Section importance multiplier
                section_name = (item.section or "").lower()
                section_mult = 1.0
                for sec_key, weight in self.section_weights.items():
                    if sec_key in section_name:
                        section_mult = max(section_mult, weight)

                # Combine base retrieval score with coverage and section weight
                new_score = (item.score * 0.5 + term_coverage * 0.5) * phrase_boost * section_mult

            scored_results.append(item.model_copy(update={"score": round(new_score, 6)}))

        scored_results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Reranked {len(results)} candidates down to {top_n}")
        return scored_results[:top_n]
