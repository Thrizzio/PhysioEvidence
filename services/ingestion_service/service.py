"""Upload validation, idempotency, and processing job orchestration."""

import hashlib
import logging
import secrets
import uuid
from datetime import datetime, timezone
from typing import Optional

from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.pipeline.document_pipeline import DocumentPipeline
from services.ingestion_service.models import DocumentStatus, IngestionJob, JobState
from services.ingestion_service.storage import LocalOriginalStore

logger = logging.getLogger(__name__)


class IngestionManager:
    def __init__(
        self,
        pipeline: Optional[DocumentPipeline] = None,
        original_store: Optional[LocalOriginalStore] = None,
        max_file_size: int = 50 * 1024 * 1024,
    ):
        self.pipeline = pipeline
        self.original_store = original_store or LocalOriginalStore("./data/originals")
        self.max_file_size = max_file_size
        self.jobs: dict[str, IngestionJob] = {}
        self.hash_to_job: dict[str, str] = {}
        self._failed_uploads = 0

    def _validate_and_hash(self, filename: str, content: bytes) -> str:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF uploads are accepted.")
        if not content:
            raise ValueError("The uploaded file is empty.")
        if len(content) > self.max_file_size:
            raise ValueError(f"The uploaded file exceeds the {self.max_file_size} byte limit.")
        return hashlib.sha256(content).hexdigest()

    def submit(self, filename: str, content: bytes) -> IngestionJob:
        try:
            content_hash = self._validate_and_hash(filename, content)
        except Exception:
            self._failed_uploads += 1
            raise

        # Parse the actual bytes before retaining them. Extension checks alone
        # are not enough to reject malformed or scanned PDFs.
        temporary_path = self.original_store.root / f".validate-{secrets.token_hex(8)}.pdf"
        try:
            temporary_path.write_bytes(content)
            extracted = PDFLoader().load(str(temporary_path))
        except Exception:
            self._failed_uploads += 1
            raise
        finally:
            temporary_path.unlink(missing_ok=True)

        existing_id = self.hash_to_job.get(content_hash)
        if existing_id:
            existing = self.jobs[existing_id]
            duplicate = existing.model_copy(update={
                "job_id": str(uuid.uuid4()),
                "state": JobState.duplicate,
                "progress": 100,
                "completed_at": datetime.now(timezone.utc),
            })
            self.jobs[duplicate.job_id] = duplicate
            return duplicate

        # A service restart should not cause a stored original to be embedded
        # again. The content hash is the durable idempotency key even when the
        # in-memory job registry has been rebuilt.
        if self.original_store.path_for(content_hash).exists():
            duplicate = IngestionJob(
                job_id=str(uuid.uuid4()),
                filename=filename,
                content_hash=content_hash,
                document_id=content_hash,
                state=JobState.duplicate,
                progress=100,
                file_size_bytes=len(content),
                completed_at=datetime.now(timezone.utc),
            )
            self.jobs[duplicate.job_id] = duplicate
            return duplicate

        path = self.original_store.save(content, content_hash)

        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            filename=filename,
            content_hash=content_hash,
            document_id=extracted.document_id,
            file_size_bytes=len(content),
        )
        self.jobs[job.job_id] = job
        self.hash_to_job[content_hash] = job.job_id
        self.process(job.job_id, str(path))
        return self.jobs[job.job_id]

    def process(self, job_id: str, path: str) -> IngestionJob:
        job = self.jobs[job_id]
        job.state = JobState.processing
        job.progress = 20
        try:
            if self.pipeline is None or self.pipeline.vector_store is None:
                raise RuntimeError("Ingestion storage is not configured.")
            chunks = self.pipeline.process(path, generate_embeddings=self.pipeline.embedder is not None)
            if self.pipeline.vector_store.count_documents() < 1 or self.pipeline.vector_store.count_chunks() < len(chunks):
                raise RuntimeError("Processing returned before storage verification succeeded.")
            job.state = JobState.succeeded
            job.progress = 100
            job.chunk_count = len(chunks)
            job.completed_at = datetime.now(timezone.utc)
        except Exception as exc:
            job.state = JobState.failed
            job.progress = 100
            job.error = str(exc)
            job.completed_at = datetime.now(timezone.utc)
            logger.exception("Ingestion job %s failed", job_id)
        return job

    def get(self, job_id: str) -> IngestionJob | None:
        return self.jobs.get(job_id)

    def record_failure(self, filename: str, content: bytes, error: str) -> IngestionJob:
        """Create a terminal job record for a per-file upload failure."""
        content_hash = hashlib.sha256(content).hexdigest()
        job = IngestionJob(
            job_id=str(uuid.uuid4()),
            filename=filename,
            content_hash=content_hash,
            state=JobState.failed,
            progress=100,
            file_size_bytes=len(content),
            error=error,
            completed_at=datetime.now(timezone.utc),
        )
        self.jobs[job.job_id] = job
        return job

    def list_documents(self) -> list[DocumentStatus]:
        return [DocumentStatus(
            document_id=job.document_id,
            filename=job.filename,
            content_hash=job.content_hash,
            state=job.state,
            chunk_count=job.chunk_count,
            file_size_bytes=job.file_size_bytes,
            error=job.error,
        ) for job in self.jobs.values() if job.state != JobState.duplicate]

    def usage(self):
        usage = self.original_store.usage()
        return usage.model_copy(update={"failed_uploads": self._failed_uploads})
