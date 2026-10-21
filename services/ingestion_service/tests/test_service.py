import hashlib
from unittest.mock import MagicMock
import pymupdf as fitz
import pytest

from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.pipeline.document_pipeline import DocumentPipeline
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.processors.section_detector import SectionDetector
from services.ingestion_service.models import IngestionJob, JobState
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


def test_failed_upload_retry(manager, monkeypatch):
    content = make_pdf()

    # Simulate failure on first process call
    original_process = manager.pipeline.process
    monkeypatch.setattr(manager.pipeline, "process", MagicMock(side_effect=RuntimeError("DB connection refused")))

    first = manager.submit("study.pdf", content)
    assert first.state == JobState.failed
    assert "DB connection refused" in first.error
    assert manager.pipeline.vector_store.count_documents() == 0

    # Restore normal pipeline and retry submission
    monkeypatch.setattr(manager.pipeline, "process", original_process)
    second = manager.submit("study.pdf", content)

    assert second.state == JobState.succeeded
    assert second.chunk_count > 0
    assert manager.pipeline.vector_store.count_documents() == 1


def test_orphaned_original_store_file_is_processed(manager):
    content = make_pdf()
    content_hash = hashlib.sha256(content).hexdigest()

    # Simulate an orphaned file already on disk (e.g. from prior failed attempt before restart)
    manager.original_store.save(content, content_hash)
    assert manager.original_store.path_for(content_hash).exists()
    assert manager.pipeline.vector_store.has_document(content_hash) is False
    assert len(manager.jobs) == 0

    # Submitting must not return duplicate; it must process and index the document
    result = manager.submit("study.pdf", content)
    assert result.state == JobState.succeeded
    assert result.chunk_count > 0
    assert manager.pipeline.vector_store.has_document(content_hash) is True
    assert manager.pipeline.vector_store.count_documents() == 1


def test_genuine_duplicate_after_restart(manager):
    content = make_pdf()
    first = manager.submit("study.pdf", content)
    assert first.state == JobState.succeeded

    # Simulate service restart: wipe in-memory registry while retaining store & DB
    manager.jobs.clear()
    manager.hash_to_job.clear()

    second = manager.submit("study.pdf", content)
    assert second.state == JobState.duplicate
    assert manager.pipeline.vector_store.count_documents() == 1


def test_database_check_failure_aborts_ingestion(manager, monkeypatch):
    content = make_pdf()
    monkeypatch.setattr(
        manager.pipeline.vector_store,
        "has_document",
        MagicMock(side_effect=RuntimeError("Database connectivity failed")),
    )

    with pytest.raises(RuntimeError, match="Database connectivity failed"):
        manager.submit("study.pdf", content)

    # Document was NOT indexed
    assert manager.pipeline.vector_store.count_documents() == 0


def test_concurrent_processing_submission_returns_existing(manager):
    content = make_pdf()
    content_hash = hashlib.sha256(content).hexdigest()

    # Simulate an in-flight processing job
    existing_job = IngestionJob(
        job_id="active-job-123",
        filename="study.pdf",
        content_hash=content_hash,
        state=JobState.processing,
        progress=40,
    )
    manager.jobs[existing_job.job_id] = existing_job
    manager.hash_to_job[content_hash] = existing_job.job_id

    result = manager.submit("study.pdf", content)
    assert result.job_id == "active-job-123"
    assert result.state == JobState.processing

