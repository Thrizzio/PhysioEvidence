import time
import statistics
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from retrieval.base import BaseRetriever
from evaluation.dataset import EvaluationSample
from evaluation.metrics import recall_at_k, mrr, ndcg_at_k, precision_at_k
from observability.tracing import trace_operation

logger = logging.getLogger(__name__)

class RetrievalExperimentResult(BaseModel):
    """
    Summary of evaluation metrics across a dataset.
    """
    retriever_name: str
    sample_count: int
    mean_recall_at_1: float
    mean_recall_at_3: float
    mean_recall_at_5: float
    mean_mrr: float
    mean_ndcg_at_5: float
    mean_latency_ms: float
    per_query_details: List[Dict[str, Any]] = Field(default_factory=list)

    def to_markdown_row(self) -> str:
        return (
            f"| {self.retriever_name:<24} | "
            f"{self.mean_recall_at_1:.3f} | "
            f"{self.mean_recall_at_3:.3f} | "
            f"{self.mean_recall_at_5:.3f} | "
            f"{self.mean_mrr:.3f} | "
            f"{self.mean_ndcg_at_5:.3f} | "
            f"{self.mean_latency_ms:.1f}ms |"
        )

class RetrievalEvaluator:
    """
    Executes benchmark queries through a retriever, measures latency,
    and calculates standard retrieval quality metrics.
    """

    def __init__(self, dataset: List[EvaluationSample]):
        self.dataset = dataset

    @trace_operation(name="retrieval_evaluation", span_type="chain")
    def evaluate(self, retriever: BaseRetriever, retriever_name: str = "Retriever", top_k: int = 5) -> RetrievalExperimentResult:
        recalls_1 = []
        recalls_3 = []
        recalls_5 = []
        mrrs = []
        ndcgs = []
        latencies = []
        query_details = []

        logger.info(f"Running evaluation for '{retriever_name}' across {len(self.dataset)} benchmark queries...")

        for sample in self.dataset:
            start_t = time.perf_counter()
            retrieved = retriever.retrieve(query=sample.query, top_k=top_k)
            latency_ms = (time.perf_counter() - start_t) * 1000.0

            retrieved_ids = [r.chunk_id for r in retrieved]
            relevant_ids = sample.relevant_chunk_ids

            r1 = recall_at_k(retrieved_ids, relevant_ids, k=1)
            r3 = recall_at_k(retrieved_ids, relevant_ids, k=3)
            r5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
            q_mrr = mrr(retrieved_ids, relevant_ids)
            q_ndcg = ndcg_at_k(retrieved_ids, relevant_ids, k=top_k)

            recalls_1.append(r1)
            recalls_3.append(r3)
            recalls_5.append(r5)
            mrrs.append(q_mrr)
            ndcgs.append(q_ndcg)
            latencies.append(latency_ms)

            query_details.append({
                "sample_id": sample.sample_id,
                "query": sample.query,
                "retrieved_ids": retrieved_ids,
                "relevant_ids": relevant_ids,
                "recall@5": round(r5, 3),
                "mrr": round(q_mrr, 3),
                "latency_ms": round(latency_ms, 2)
            })

        result = RetrievalExperimentResult(
            retriever_name=retriever_name,
            sample_count=len(self.dataset),
            mean_recall_at_1=round(statistics.mean(recalls_1), 4),
            mean_recall_at_3=round(statistics.mean(recalls_3), 4),
            mean_recall_at_5=round(statistics.mean(recalls_5), 4),
            mean_mrr=round(statistics.mean(mrrs), 4),
            mean_ndcg_at_5=round(statistics.mean(ndcgs), 4),
            mean_latency_ms=round(statistics.mean(latencies), 2),
            per_query_details=query_details
        )

        logger.info(
            f"Evaluation finished for '{retriever_name}': "
            f"Recall@5={result.mean_recall_at_5}, MRR={result.mean_mrr}, Latency={result.mean_latency_ms}ms"
        )
        return result

    @staticmethod
    def format_comparison_table(results: List[RetrievalExperimentResult]) -> str:
        """
        Formats multiple experiment results into a clean markdown table.
        """
        lines = [
            "| Strategy                 | R@1   | R@3   | R@5   | MRR   | nDCG@5 | Latency |",
            "|:-------------------------|:------|:------|:------|:------|:-------|:--------|"
        ]
        for res in results:
            lines.append(res.to_markdown_row())
        return "\n".join(lines)
