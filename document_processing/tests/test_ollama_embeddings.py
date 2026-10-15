import json
import math
import os
from unittest.mock import MagicMock, patch
import pytest
import requests

from document_processing.embeddings.ollama_embeddings import (
    OllamaEmbeddingProvider,
    OllamaConnectionError,
    OllamaTimeoutError,
    OllamaModelNotFoundError,
    OllamaAPIError,
    OllamaResponseError,
)
from document_processing.models.chunk import Chunk, ChunkMetadata
from document_processing.models.document import ExtractedDocument, Page, DocumentMetadata
from document_processing.pipeline.document_pipeline import DocumentPipeline
from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.processors.section_detector import SectionDetector
from document_processing.processors.chunker import SectionAwareChunker
from storage.in_memory_store import InMemoryVectorStore
from retrieval.vector_retriever import VectorRetriever
from retrieval.hybrid_retriever import HybridRetriever
import pymupdf as fitz


# ---------------------------------------------------------------------------
# Unit tests: Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_query():
    provider = OllamaEmbeddingProvider()
    
    # Standard query
    q = "What quadriceps symmetry index is needed?"
    formatted = provider.preprocess_query(q)
    assert formatted == "task: search result | query: What quadriceps symmetry index is needed?"

    # Already formatted query - should not duplicate
    already = "task: search result | query: What quadriceps symmetry index is needed?"
    assert provider.preprocess_query(already) == already

    # Case insensitive task prefix
    already_upper = "TASK: question answering | query: test"
    assert provider.preprocess_query(already_upper) == already_upper

    # Empty query raises ValueError
    with pytest.raises(ValueError, match="Query text cannot be empty"):
        provider.preprocess_query("   ")


def test_preprocess_document():
    provider = OllamaEmbeddingProvider()

    # Document with title
    doc = provider.preprocess_document("Patient achieved 92% QSI at 6 months.", title="ACL Reconstruction Study")
    assert doc == "title: ACL Reconstruction Study | text: Patient achieved 92% QSI at 6 months."

    # Document without title
    doc_no_title = provider.preprocess_document("Patient achieved 92% QSI at 6 months.", title=None)
    assert doc_no_title == "title: none | text: Patient achieved 92% QSI at 6 months."

    # Document already containing title prefix - should not duplicate
    already = "title: Existing Title | text: Some content"
    assert provider.preprocess_document(already) == already

    # Empty content raises ValueError
    with pytest.raises(ValueError, match="content cannot be empty"):
        provider.preprocess_document("  ")


# ---------------------------------------------------------------------------
# Unit tests: Mocked Ollama Embedding API
# ---------------------------------------------------------------------------

def test_successful_single_text_embedding_mocked():
    provider = OllamaEmbeddingProvider(dimensions=768)
    dummy_vec = [0.01 * (i % 10) for i in range(768)]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "embeddinggemma-2:270m",
        "embeddings": [dummy_vec],
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        vec = provider.embed_query("stroke rehabilitation")
        assert len(vec) == 768
        assert vec == dummy_vec

        # Confirm request went to /api/embed and never to chat or completion endpoints
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:11434/api/embed"
        assert kwargs["json"]["model"] == "embeddinggemma-2:270m"
        assert kwargs["json"]["input"] == ["task: search result | query: stroke rehabilitation"]
        assert "/chat" not in args[0]
        assert "/generate" not in args[0]


def test_successful_batch_embedding_mocked():
    provider = OllamaEmbeddingProvider(dimensions=768, batch_size=2)
    dummy_vec_1 = [0.1] * 768
    dummy_vec_2 = [0.2] * 768
    dummy_vec_3 = [0.3] * 768

    chunks = [
        Chunk(chunk_id="c1", document_id="d1", chunk_index=0, content="Content 1", metadata=ChunkMetadata(document_title="Doc 1")),
        Chunk(chunk_id="c2", document_id="d1", chunk_index=1, content="Content 2", metadata=ChunkMetadata(document_title="Doc 1")),
        Chunk(chunk_id="c3", document_id="d1", chunk_index=2, content="Content 3", metadata=ChunkMetadata(document_title="Doc 1")),
    ]

    mock_resp_1 = MagicMock(status_code=200)
    mock_resp_1.json.return_value = {"embeddings": [dummy_vec_1, dummy_vec_2]}
    mock_resp_2 = MagicMock(status_code=200)
    mock_resp_2.json.return_value = {"embeddings": [dummy_vec_3]}

    with patch("requests.post", side_effect=[mock_resp_1, mock_resp_2]) as mock_post:
        embedded_chunks = provider.embed_chunks(chunks)
        assert len(embedded_chunks) == 3
        assert embedded_chunks[0].embedding == dummy_vec_1
        assert embedded_chunks[1].embedding == dummy_vec_2
        assert embedded_chunks[2].embedding == dummy_vec_3
        assert mock_post.call_count == 2


def test_empty_input_handling():
    provider = OllamaEmbeddingProvider()
    with patch("requests.post") as mock_post:
        assert provider.embed_texts([]) == []
        assert provider.embed_chunks([]) == []
        mock_post.assert_not_called()


def test_malformed_responses_and_invalid_vectors():
    provider = OllamaEmbeddingProvider(dimensions=768)

    # 1. Missing 'embeddings' field
    bad_resp = MagicMock(status_code=200)
    bad_resp.json.return_value = {"wrong_key": [[0.1] * 768]}
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(OllamaResponseError, match="expected dict with 'embeddings' key"):
            provider.embed_query("test")

    # 2. Non-list embeddings
    bad_resp.json.return_value = {"embeddings": "not a list"}
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(OllamaResponseError, match="Expected 'embeddings' to be a list"):
            provider.embed_query("test")

    # 3. Vector count mismatch
    bad_resp.json.return_value = {"embeddings": []}
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(OllamaResponseError, match="Input count mismatch"):
            provider.embed_query("test")

    # 4. Dimension mismatch
    bad_resp.json.return_value = {"embeddings": [[0.1] * 512]}
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(OllamaResponseError, match="Dimension mismatch"):
            provider.embed_query("test")

    # 5. Non-finite values (NaN / Inf)
    bad_resp.json.return_value = {"embeddings": [[float("nan")] * 768]}
    with patch("requests.post", return_value=bad_resp):
        with pytest.raises(OllamaResponseError, match="Non-finite or non-numeric value"):
            provider.embed_query("test")

    # 6. Invalid JSON body
    bad_json_resp = MagicMock(status_code=200)
    bad_json_resp.json.side_effect = ValueError("Invalid JSON")
    bad_json_resp.text = "Internal Server Error HTML"
    with patch("requests.post", return_value=bad_json_resp):
        with pytest.raises(OllamaResponseError, match="could not be parsed as JSON"):
            provider.embed_query("test")


def test_connection_failure_and_missing_model_errors():
    provider = OllamaEmbeddingProvider()

    # Connection error
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        with pytest.raises(OllamaConnectionError, match="Failed to connect to Ollama"):
            provider.embed_query("test")

    # Timeout error
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Read timed out")):
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            provider.embed_query("test")

    # Model not found (404)
    resp_404 = MagicMock(status_code=404)
    resp_404.json.return_value = {"error": "model 'embeddinggemma-2:270m' not found"}
    with patch("requests.post", return_value=resp_404):
        with pytest.raises(OllamaModelNotFoundError, match="was not found on server"):
            provider.embed_query("test")

    # General HTTP error (500)
    resp_500 = MagicMock(status_code=500)
    resp_500.json.return_value = {"error": "cuda out of memory"}
    with patch("requests.post", return_value=resp_500):
        with pytest.raises(OllamaAPIError, match="Ollama embedding request failed"):
            provider.embed_query("test")


# ---------------------------------------------------------------------------
# Integration with Ingestion and Retrieval Pipeline
# ---------------------------------------------------------------------------

def test_pipeline_ingestion_and_retrieval_with_ollama(tmp_path):
    # Create a synthetic clinical research paper PDF
    pdf_path = tmp_path / "clinical_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Title: Quadriceps Symmetry Index After Anterior Cruciate Ligament Reconstruction.\n\n"
        "Abstract: Return to sport guidelines require a quadriceps symmetry index (QSI) of at least 90 percent. "
        "Athletes achieving 90 percent QSI exhibited significantly lower rates of secondary ACL rupture at two years follow-up. "
        "Limb symmetry testing remains essential in progressive rehabilitation programs."
    )
    doc.save(str(pdf_path))
    doc.close()

    # Mock embedder that returns consistent 768-dim vectors
    embedder = OllamaEmbeddingProvider(dimensions=768)
    
    def fake_embed(texts):
        # Generate deterministic vector based on text content
        vectors = []
        for t in texts:
            seed = sum(ord(c) for c in t) % 100
            vec = [float(seed) / 100.0] * 768
            # Normalize vector
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors

    with patch.object(embedder, "_call_api", side_effect=fake_embed):
        store = InMemoryVectorStore()
        pipeline = DocumentPipeline(
            loader=PDFLoader(),
            section_detector=SectionDetector(),
            chunker=SectionAwareChunker(),
            embedder=embedder,
            vector_store=store,
        )

        # 1. Ingest PDF
        chunks = pipeline.process(str(pdf_path), generate_embeddings=True)
        assert len(chunks) > 0
        assert all(c.embedding is not None for c in chunks)
        assert len(chunks[0].embedding) == 768
        assert store.count_chunks() == len(chunks)

        # 2. Retrieve via VectorRetriever
        retriever = VectorRetriever(vector_store=store, embedder=embedder)
        results = retriever.retrieve("quadriceps symmetry index ACL", top_k=2)
        assert len(results) > 0
        assert results[0].content is not None
        assert results[0].score > 0.0

        # 3. Retrieve via HybridRetriever
        hybrid = HybridRetriever(vector_store=store, embedder=embedder)
        hybrid_results = hybrid.retrieve("quadriceps symmetry index ACL", top_k=2)
        assert len(hybrid_results) > 0


# ---------------------------------------------------------------------------
# Live Integration Test (Executed when local Ollama daemon is running)
# ---------------------------------------------------------------------------

def test_live_ollama_integration():
    """Calls the local Ollama instance if running with embeddinggemma-2:270m."""
    try:
        check = requests.get("http://localhost:11434/api/tags", timeout=2)
        if check.status_code != 200:
            pytest.skip("Local Ollama daemon is not responding on http://localhost:11434")
        models = [m["name"] for m in check.json().get("models", [])]
        if not any("embeddinggemma-2:270m" in m for m in models):
            pytest.skip("embeddinggemma-2:270m is not installed in local Ollama instance")
    except Exception as exc:
        pytest.skip(f"Ollama server not reachable: {exc}")

    provider = OllamaEmbeddingProvider(model_name="embeddinggemma-2:270m")

    # 1. Single query embedding
    query_vec = provider.embed_query("What quadriceps symmetry index is needed after ACL reconstruction?")
    assert len(query_vec) == 768
    assert all(math.isfinite(x) for x in query_vec)
    l2_query = math.sqrt(sum(x * x for x in query_vec))
    assert math.isclose(l2_query, 1.0, rel_tol=1e-4)

    # 2. Batch chunk embedding
    chunks = [
        Chunk(
            chunk_id="live_c1",
            document_id="live_d1",
            chunk_index=0,
            content="Quadriceps symmetry index above 90% significantly reduces reinjury risk.",
            metadata=ChunkMetadata(document_title="ACL Clinical Guide"),
        ),
        Chunk(
            chunk_id="live_c2",
            document_id="live_d1",
            chunk_index=1,
            content="Progressive resistance training begins early post-operative.",
            metadata=ChunkMetadata(document_title="ACL Clinical Guide"),
        ),
    ]
    embedded = provider.embed_chunks(chunks)
    assert len(embedded) == 2
    assert len(embedded[0].embedding) == 768
    assert len(embedded[1].embedding) == 768
    assert all(math.isfinite(x) for x in embedded[0].embedding)
    assert all(math.isfinite(x) for x in embedded[1].embedding)
