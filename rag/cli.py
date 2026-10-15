"""
Command-line interface for asking clinical questions against the PhysioEvidence corpus.
Usage:
    python -m rag.cli --question "What quadriceps symmetry index is needed after ACL reconstruction?"
"""
import argparse
import sys
from dotenv import load_dotenv

from document_processing.utils.logging import setup_logging
from services.common import build_embedder
from storage.db import get_connection
from storage.postgres_store import PostgresVectorStore
from storage.base import FilterCriteria
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.llm import GeminiLLMProvider
from generation.synthesizer import EvidenceSynthesizer
from rag.pipeline import PhysioEvidenceRAG

def main():
    parser = argparse.ArgumentParser(description="PhysioEvidence Clinical Evidence Assistant.")
    parser.add_argument("--question", "-q", required=True, help="Clinical research question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks to retrieve")
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid retrieval (dense + sparse RRF) instead of pure vector")
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder/heuristic reranker")
    parser.add_argument("--section", help="Filter evidence by specific paper section (e.g. Results, Methods)")
    parser.add_argument("--min-year", type=int, help="Filter by minimum publication year")
    parser.add_argument("--embedding-provider", choices=["gemini", "local", "ollama"], default=None, help="Embedding provider; defaults to EMBEDDING_PROVIDER or gemini")
    
    args = parser.parse_args()

    load_dotenv()
    setup_logging()

    if args.embedding_provider:
        import os
        os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider

    # Initialize components
    try:
        embedder = build_embedder()
    except Exception as e:
        print(f"Error initializing embedder: {e}")
        sys.exit(1)

    try:
        vector_store = PostgresVectorStore()
    except Exception as e:
        print(f"Error connecting to PostgreSQL vector store: {e}")
        sys.exit(1)

    if args.hybrid:
        retriever = HybridRetriever(vector_store=vector_store, embedder=embedder)
    else:
        retriever = VectorRetriever(vector_store=vector_store, embedder=embedder)

    reranker = CrossEncoderReranker() if args.rerank else None

    try:
        llm = GeminiLLMProvider()
    except ValueError as e:
        print(f"Error: {e}. Set GEMINI_API_KEY in your environment.")
        sys.exit(1)

    synthesizer = EvidenceSynthesizer(llm=llm)
    rag = PhysioEvidenceRAG(retriever=retriever, synthesizer=synthesizer, reranker=reranker)

    # Filter criteria
    filters = None
    if args.section or args.min_year:
        filters = FilterCriteria(
            sections=[args.section] if args.section else None,
            min_year=args.min_year
        )

    print(f"\nSearching scientific literature for: '{args.question}'...")
    answer = rag.ask(question=args.question, top_k=args.top_k, filter_criteria=filters)

    print("\n" + "=" * 80)
    print("PHYSIOEVIDENCE CLINICAL ANSWER")
    print("=" * 80)
    print(answer.format_for_display())
    print("\n" + "-" * 80)
    print(f"Total Latency: {answer.latency_ms}ms | Sources Retrieved: {len(answer.sources)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
