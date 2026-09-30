"""
Standalone benchmark runner for comparing retrieval strategies on the curated physiotherapy dataset.
Usage:
    python -m evaluation.run_benchmark
"""
import sys
import logging
from storage.in_memory_store import InMemoryVectorStore
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from evaluation.dataset import get_curated_evaluation_dataset, get_benchmark_corpus_chunks
from evaluation.retrieval_evaluator import RetrievalEvaluator

logging.basicConfig(level=logging.WARNING)

def mock_query_embedder(query: str):
    q = query.lower()
    if "acl" in q or "quadriceps" in q:
        return [0.85, 0.15, 0.05, 0.0]
    elif "patellofemoral" in q or "hip" in q:
        return [0.10, 0.85, 0.10, 0.0]
    elif "achilles" in q or "alfredson" in q or "resistance" in q:
        return [0.05, 0.15, 0.85, 0.0]
    elif "low back" in q or "motor control" in q:
        return [0.10, 0.10, 0.10, 0.85]
    elif "rotator cuff" in q or "isometric" in q:
        return [0.30, 0.10, 0.70, 0.10]
    elif "stroke" in q or "cimt" in q:
        return [0.05, 0.05, 0.05, 0.90]
    return [0.25, 0.25, 0.25, 0.25]

class RerankingRetriever:
    """Wrapper that composes a retriever with a reranker."""
    def __init__(self, base_retriever, reranker, pool_k=10):
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.pool_k = pool_k

    def retrieve(self, query: str, top_k: int = 5, filter_criteria=None):
        candidates = self.base_retriever.retrieve(query, top_k=self.pool_k, filter_criteria=filter_criteria)
        return self.reranker.rerank(query, candidates, top_n=top_k)

def main():
    print("=" * 80)
    print("PhysioEvidence - Retrieval Strategy Benchmark")
    print("Evaluating strategies across curated clinical physiotherapy queries")
    print("=" * 80)

    dataset = get_curated_evaluation_dataset()
    corpus_chunks = get_benchmark_corpus_chunks()

    store = InMemoryVectorStore()
    for doc_id in {"doc_acl", "doc_pfps", "doc_achilles", "doc_lbp", "doc_rc", "doc_stroke", "doc_distractor"}:
        store.save_document(StructuredDocument(
            document_id=doc_id,
            source_path=f"/{doc_id}.pdf",
            filename=f"{doc_id}.pdf",
            metadata=DocumentMetadata(title=doc_id),
            sections=[]
        ))

    chunks = []
    for c in corpus_chunks:
        chunks.append(Chunk(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            chunk_index=0,
            content=c["content"],
            metadata=ChunkMetadata(section=c["section"], document_title=c["title"]),
            embedding=c["embedding"]
        ))
    store.save_chunks(chunks)

    vector_retriever = VectorRetriever(vector_store=store, embedder=mock_query_embedder)
    hybrid_retriever = HybridRetriever(vector_store=store, embedder=mock_query_embedder)
    hybrid_rerank_retriever = RerankingRetriever(
        base_retriever=hybrid_retriever,
        reranker=CrossEncoderReranker()
    )

    evaluator = RetrievalEvaluator(dataset=dataset)

    res_vector = evaluator.evaluate(vector_retriever, retriever_name="1. Dense Vector Only", top_k=5)
    res_hybrid = evaluator.evaluate(hybrid_retriever, retriever_name="2. Hybrid (Vector+RRF)", top_k=5)
    res_rerank = evaluator.evaluate(hybrid_rerank_retriever, retriever_name="3. Hybrid + Reranker", top_k=5)

    print("\nBenchmark Results:")
    print(RetrievalEvaluator.format_comparison_table([res_vector, res_hybrid, res_rerank]))
    print("=" * 80)

if __name__ == "__main__":
    main()
