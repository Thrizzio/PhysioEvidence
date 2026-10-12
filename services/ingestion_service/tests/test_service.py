import pymupdf as fitz
import pytest

from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.pipeline.document_pipeline import DocumentPipeline
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.processors.section_detector import SectionDetector
from services.ingestion_service.service import IngestionManager
from services.ingestion_service.storage import LocalOriginalStore
from storage.in_memory_store import InMemoryVectorStore


def make_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text(
        (50, 50),
        "Abstract\nThis physiotherapy study contains enough extractable text for validation. "
        "The result is retained with a page reference for evidence inspection.",
    )
    content = document.tobytes()
    document.close()
    return content


@pytest.fixture
def manager(tmp_path):
    store = InMemoryVectorStore()
    pipeline = DocumentPipeline(
        loader=PDFLoader(),
        section_detector=SectionDetector(),
        chunker=SectionAwareChunker(min_chunk_size=10),
        vector_store=store,
    )
    return IngestionManager(
        pipeline=pipeline,
        original_store=LocalOriginalStore(str(tmp_path / "originals")),
    )


def test_upload_is_validated_indexed_and_idempotent(manager):
    content = make_pdf()
    first = manager.submit("study.pdf", content)
    second = manager.submit("renamed-study.pdf", content)

    assert first.state.value == "succeeded"
    assert first.document_id
    assert first.chunk_count > 0
    assert second.state.value == "duplicate"
    assert manager.pipeline.vector_store.count_documents() == 1


def test_invalid_uploads_are_visible_failures(manager):
    with pytest.raises(ValueError, match="Only PDF"):
        manager.submit("study.txt", b"not a pdf")
    with pytest.raises(ValueError, match="empty"):
        manager.submit("empty.pdf", b"")
    assert manager.usage().failed_uploads == 2
