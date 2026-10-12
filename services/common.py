"""Shared service configuration and RAG construction.

The service packages remain separate at the HTTP boundary while sharing the
canonical processing, storage, retrieval, and generation abstractions.
"""

import os
from typing import Optional

from dotenv import load_dotenv

from document_processing.embeddings.base import EmbeddingProvider
from document_processing.embeddings.gemini_embeddings import GeminiEmbeddingProvider
from document_processing.embeddings.local_embeddings import LocalEmbeddingProvider
from generation.llm import GeminiLLMProvider
from generation.synthesizer import EvidenceSynthesizer
from rag.pipeline import PhysioEvidenceRAG
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from storage.postgres_store import PostgresVectorStore

load_dotenv()


def build_embedder() -> EmbeddingProvider:
    provider = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
    if provider == "local":
        return LocalEmbeddingProvider(
            model_name=os.getenv("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m"),
            dimensions=int(os.getenv("LOCAL_EMBEDDING_DIMENSIONS", "768")),
        )
    if provider != "gemini":
        raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")
    return GeminiEmbeddingProvider(model_name=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"))


def build_rag(*, hybrid: bool = True) -> PhysioEvidenceRAG:
    """Build the production dependency graph lazily at service startup."""
    store = PostgresVectorStore()
    embedder = build_embedder()
    retriever = (
        HybridRetriever(store, embedder)
        if hybrid
        else VectorRetriever(store, embedder)
    )
    synthesizer = EvidenceSynthesizer(GeminiLLMProvider())
    return PhysioEvidenceRAG(retriever=retriever, synthesizer=synthesizer)


def configured_origins(name: str) -> list[str]:
    value = os.getenv(name, "")
    return [origin.strip() for origin in value.split(",") if origin.strip()]
