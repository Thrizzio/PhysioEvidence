import pytest
from unittest.mock import MagicMock
from pathlib import Path
from document_processing.pipeline.document_pipeline import DocumentPipeline, DocumentProcessingError
from document_processing.models.document import ExtractedDocument, Page, DocumentMetadata, Section, StructuredDocument
from document_processing.models.chunk import Chunk, ChunkMetadata
from storage.in_memory_store import InMemoryVectorStore

@pytest.fixture
def mock_pipeline_components():
    loader = MagicMock()
    section_detector = MagicMock()
    chunker = MagicMock()
    embedder = MagicMock()
    vector_store = InMemoryVectorStore()

    extracted_doc = ExtractedDocument(
        document_id="doc_pipe_1",
        source_path="/path/test.pdf",
        filename="test.pdf",
        metadata=DocumentMetadata(title="Test Paper"),
        pages=[Page(page_number=1, text="Sample text")]
    )
    loader.load.return_value = extracted_doc

    section_detector.detect_sections.return_value = [
        Section(name="Introduction", content="Sample text", page_start=1, page_end=1)
    ]

    chunk = Chunk(
        chunk_id="chunk_pipe_1",
        document_id="doc_pipe_1",
        chunk_index=0,
        content="Sample text",
        metadata=ChunkMetadata(section="Introduction", page_start=1, page_end=1),
        embedding=[0.1] * 768
    )
    chunker.chunk.return_value = [chunk]
    embedder.embed_chunks.side_effect = lambda chunks: chunks

    pipeline = DocumentPipeline(
        loader=loader,
        section_detector=section_detector,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store
    )
    return pipeline, vector_store, loader

def test_pipeline_end_to_end_with_store(mock_pipeline_components):
    pipeline, vector_store, loader = mock_pipeline_components
    chunks = pipeline.process("/path/test.pdf", generate_embeddings=True)

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "chunk_pipe_1"
    assert vector_store.count_documents() == 1
    assert vector_store.count_chunks() == 1
    assert vector_store.get_chunk("chunk_pipe_1") is not None

def test_pipeline_error_wrapping(mock_pipeline_components):
    pipeline, _, loader = mock_pipeline_components
    loader.load.side_effect = RuntimeError("Corrupted PDF file")

    with pytest.raises(DocumentProcessingError) as exc_info:
        pipeline.process("/path/corrupted.pdf")

    assert exc_info.value.stage == "extraction"
    assert "Corrupted PDF file" in str(exc_info.value)

def test_pipeline_process_directory(tmp_path, mock_pipeline_components):
    pipeline, vector_store, loader = mock_pipeline_components

    # Create dummy pdf files in tmp_path
    pdf1 = tmp_path / "paper1.pdf"
    pdf1.write_text("dummy")
    pdf2 = tmp_path / "paper2.pdf"
    pdf2.write_text("dummy")

    summary = pipeline.process_directory(str(tmp_path), generate_embeddings=False)

    assert summary["total_files"] == 2
    assert len(summary["successful_files"]) == 2
    assert len(summary["failed_files"]) == 0
    assert summary["total_chunks"] == 2
