import pytest
from evaluation.metrics import recall_at_k, precision_at_k, mrr, ndcg_at_k
from evaluation.dataset import get_curated_evaluation_dataset, get_benchmark_corpus_chunks
from evaluation.retrieval_evaluator import RetrievalEvaluator
from storage.in_memory_store import InMemoryVectorStore
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever

def test_metrics_calculation():
    retrieved = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    relevant = ["doc_B", "doc_D"]

    # Recall@K
    assert recall_at_k(retrieved, relevant, k=1) == 0.0  # doc_A is not in relevant
    assert recall_at_k(retrieved, relevant, k=2) == 0.5  # doc_B is in relevant (1 of 2)
    assert recall_at_k(retrieved, relevant, k=4) == 1.0  # both doc_B and doc_D found (2 of 2)

    # Precision@K
    assert precision_at_k(retrieved, relevant, k=2) == 0.5   # 1 out of 2
    assert precision_at_k(retrieved, relevant, k=4) == 0.5   # 2 out of 4

    # MRR (first relevant is doc_B at rank 2)
    assert mrr(retrieved, relevant) == 0.5

    # nDCG@K
    ndcg = ndcg_at_k(retrieved, relevant, k=5)
    assert 0.0 < ndcg <= 1.0

def test_comparative_retrieval_experiment():
    dataset = get_curated_evaluation_dataset()
    corpus_chunks = get_benchmark_corpus_chunks()

    # Populate store with benchmark corpus
    store = InMemoryVectorStore()
    
    # Save dummy docs
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

    # Mock embedder that maps query keywords to corresponding topic vectors
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

    vector_retriever = VectorRetriever(vector_store=store, embedder=mock_query_embedder)
    hybrid_retriever = HybridRetriever(vector_store=store, embedder=mock_query_embedder)

    evaluator = RetrievalEvaluator(dataset=dataset)

    vector_res = evaluator.evaluate(vector_retriever, retriever_name="Vector Only", top_k=5)
    hybrid_res = evaluator.evaluate(hybrid_retriever, retriever_name="Hybrid (Vector + RRF)", top_k=5)

    assert vector_res.sample_count == 6
    assert hybrid_res.sample_count == 6
    assert vector_res.mean_recall_at_5 > 0.0
    assert hybrid_res.mean_recall_at_5 > 0.0

    table = RetrievalEvaluator.format_comparison_table([vector_res, hybrid_res])
    assert "| Strategy" in table
    assert "Vector Only" in table
    assert "Hybrid (Vector + RRF)" in table
