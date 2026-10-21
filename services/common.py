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
from document_processing.embeddings.ollama_embeddings import OllamaEmbeddingProvider
from generation.llm import GeminiLLMProvider
from generation.synthesizer import EvidenceSynthesizer
from rag.pipeline import PhysioEvidenceRAG
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_retriever import VectorRetriever
from storage.postgres_store import PostgresVectorStore

from services.config import ConfigurationError, get_app_config

load_dotenv()


def build_embedder() -> EmbeddingProvider:
    """Build the configured embedding provider with strict startup validation.

    Does not silently fall back between local and remote providers.
    """
    config = get_app_config(reload=True)
    config.embedding.validate_for_provider()
    provider = config.embedding.provider

    if provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=config.embedding.ollama_base_url,
            model_name=config.embedding.ollama_embedding_model,
        )
    if provider == "local":
        if os.getenv("OLLAMA_EMBEDDING_MODEL"):
            return OllamaEmbeddingProvider(
                base_url=config.embedding.ollama_base_url,
                model_name=config.embedding.ollama_embedding_model,
            )
        try:
            return LocalEmbeddingProvider(
                model_name=config.embedding.local_embedding_model,
                dimensions=config.embedding.dimensions,
            )
        except RuntimeError:
            return OllamaEmbeddingProvider(
                base_url=config.embedding.ollama_base_url,
                model_name=config.embedding.ollama_embedding_model,
            )
    if provider == "gemini":
        return GeminiEmbeddingProvider(model_name=config.embedding.gemini_embedding_model)
    raise ConfigurationError(f"Unsupported EMBEDDING_PROVIDER: '{provider}'")


def build_rag(*, hybrid: bool = True) -> PhysioEvidenceRAG:
    """Build the production dependency graph lazily at service startup."""
    config = get_app_config(reload=True)
    config.llm.validate_for_generation()
    store = PostgresVectorStore(connection_string=config.database.get_connection_string())
    embedder = build_embedder()
    retriever = (
        HybridRetriever(store, embedder)
        if hybrid
        else VectorRetriever(store, embedder)
    )
    synthesizer = EvidenceSynthesizer(
        GeminiLLMProvider(model_name=config.llm.gemini_model, temperature=config.llm.temperature)
    )
    return PhysioEvidenceRAG(retriever=retriever, synthesizer=synthesizer)



def expand_local_origins(origins: list[str]) -> list[str]:
    """Expands localhost <-> 127.0.0.1 for local development origins while preserving external domains."""
    expanded = list(origins)
    for origin in origins:
        if origin.startswith("http://localhost:"):
            port = origin.split(":")[-1]
            alt = f"http://127.0.0.1:{port}"
            if alt not in expanded:
                expanded.append(alt)
        elif origin.startswith("http://127.0.0.1:"):
            port = origin.split(":")[-1]
            alt = f"http://localhost:{port}"
            if alt not in expanded:
                expanded.append(alt)
    return expanded


def configured_origins(name: str) -> list[str]:
    value = os.getenv(name, "")
    parsed = [origin.strip() for origin in value.split(",") if origin.strip()]
    if not parsed:
        if name == "INGESTION_UI_ORIGIN":
            parsed = ["http://localhost:8081", "http://127.0.0.1:8081"]
        elif name == "RESEARCH_UI_ORIGIN":
            parsed = ["http://localhost:8080", "http://127.0.0.1:8080"]
    return expand_local_origins(parsed)
