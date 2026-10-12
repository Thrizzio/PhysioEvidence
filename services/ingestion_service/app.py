"""Private admin API for uploading and monitoring documents."""

import logging
import hmac
import os

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.pipeline.document_pipeline import DocumentPipeline
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.processors.section_detector import SectionDetector
from services.common import build_embedder, configured_origins
from services.ingestion_service.models import IngestionJob
from services.ingestion_service.service import IngestionManager
from services.ingestion_service.storage import LocalOriginalStore
from storage.postgres_store import PostgresVectorStore

logger = logging.getLogger(__name__)


def build_manager() -> IngestionManager:
    store = PostgresVectorStore()
    pipeline = DocumentPipeline(
        loader=PDFLoader(),
        section_detector=SectionDetector(),
        chunker=SectionAwareChunker(),
        embedder=build_embedder(),
        vector_store=store,
    )
    quota = os.getenv("INGESTION_QUOTA_BYTES")
    return IngestionManager(
        pipeline=pipeline,
        original_store=LocalOriginalStore(
            os.getenv("INGESTION_STORAGE_ROOT", "./data/originals"),
            quota_bytes=int(quota) if quota else None,
        ),
        max_file_size=int(os.getenv("INGESTION_MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024))),
    )


def create_app(manager: IngestionManager | None = None) -> FastAPI:
    app = FastAPI(title="PhysioEvidence Private Ingestion API", version="1.0")
    origins = configured_origins("INGESTION_UI_ORIGIN")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type"],
        )
    app.state.manager = manager

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        expected = os.getenv("INGESTION_ADMIN_TOKEN")
        if not expected:
            raise HTTPException(status_code=503, detail="INGESTION_ADMIN_TOKEN is not configured.")
        supplied = authorization.removeprefix("Bearer ") if authorization else ""
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="Admin authorization required.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "ingestion-service", "configured": app.state.manager is not None}

    @app.post("/api/uploads", response_model=list[IngestionJob], dependencies=[Depends(require_admin)])
    async def upload(files: list[UploadFile] = File(...)) -> list[IngestionJob]:
        if app.state.manager is None:
            try:
                app.state.manager = build_manager()
            except Exception as exc:
                logger.exception("Ingestion service dependency initialization failed")
                raise HTTPException(status_code=503, detail="Ingestion service is not configured.") from exc
        results = []
        for upload_file in files:
            content = b""
            try:
                content = await upload_file.read()
                results.append(app.state.manager.submit(upload_file.filename or "upload.pdf", content))
            except Exception as exc:
                filename = upload_file.filename or "upload.pdf"
                results.append(app.state.manager.record_failure(filename, content, str(exc)))
        return results

    @app.get("/api/jobs/{job_id}", response_model=IngestionJob, dependencies=[Depends(require_admin)])
    def job_status(job_id: str) -> IngestionJob:
        if app.state.manager is None or (job := app.state.manager.get(job_id)) is None:
            raise HTTPException(status_code=404, detail="Ingestion job not found.")
        return job

    @app.get("/api/documents", dependencies=[Depends(require_admin)])
    def documents() -> dict:
        if app.state.manager is None:
            return {"documents": [], "usage": None}
        return {"documents": app.state.manager.list_documents(), "usage": app.state.manager.usage()}

    return app


app = create_app()
