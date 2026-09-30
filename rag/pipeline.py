import logging
import time
from typing import Optional
from retrieval.base import BaseRetriever
from retrieval.reranker import BaseReranker
from generation.synthesizer import EvidenceSynthesizer, EvidenceAnswer
from storage.base import FilterCriteria
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

class PhysioEvidenceRAG:
    """
    End-to-end RAG orchestrator for PhysioEvidence.
    Coordinates evidence retrieval, optional neural/heuristic reranking,
    and source-grounded answer generation with verified citations.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        synthesizer: EvidenceSynthesizer,
        reranker: Optional[BaseReranker] = None
    ):
        self.retriever = retriever
        self.synthesizer = synthesizer
        self.reranker = reranker

    @trace_operation(name="physio_evidence_rag", span_type="chain")
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> EvidenceAnswer:
        """
        Executes the full RAG pipeline:
        Question -> Retriever -> (Optional Reranker) -> Context Assembly -> LLM -> Grounded Evidence Answer.
        """
        start_t = time.perf_counter()
        logger.info(f"Processing clinical question: '{question}'")

        # 1. Retrieve candidates
        candidates = self.retriever.retrieve(
            query=question,
            top_k=top_k * 2 if self.reranker else top_k,
            filter_criteria=filter_criteria
        )

        # 2. Rerank if configured
        if self.reranker is not None and candidates:
            candidates = self.reranker.rerank(
                query=question,
                results=candidates,
                top_n=top_k
            )

        # 3. Synthesize grounded answer
        answer = self.synthesizer.synthesize(
            question=question,
            search_results=candidates
        )

        total_latency = (time.perf_counter() - start_t) * 1000.0
        answer.latency_ms = round(total_latency, 2)
        logger.info(f"RAG answer generated in {total_latency:.1f}ms")
        return answer
