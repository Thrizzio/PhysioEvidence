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
from services.config import get_app_config
from services.ingestion_service.models import IngestionJob
from services.ingestion_service.service import IngestionManager
from services.ingestion_service.storage import LocalOriginalStore
from storage.postgres_store import PostgresVectorStore

logger = logging.getLogger(__name__)


def build_manager() -> IngestionManager:
    config = get_app_config()
    store = PostgresVectorStore(connection_string=config.database.get_connection_string())
    pipeline = DocumentPipeline(
        loader=PDFLoader(),
        section_detector=SectionDetector(),
        chunker=SectionAwareChunker(),
        embedder=build_embedder(),
        vector_store=store,
    )
    quota = config.ingestion.quota_bytes
    return IngestionManager(
        pipeline=pipeline,
        original_store=LocalOriginalStore(
            config.ingestion.storage_root,
            quota_bytes=quota,
        ),
        max_file_size=config.ingestion.max_file_size_bytes,
    )


def create_app(manager: IngestionManager | None = None) -> FastAPI:
    app = FastAPI(title="PhysioEvidence Private Ingestion API", version="1.0")
    origins = configured_origins("INGESTION_UI_ORIGIN")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
            allow_credentials=True,
        )
    app.state.manager = manager

    def require_admin(authorization: str | None = Header(default=None)) -> None:
        expected = (os.getenv("INGESTION_ADMIN_TOKEN") or "").strip()
        if not expected:
            raise HTTPException(status_code=503, detail="INGESTION_ADMIN_TOKEN is not configured.")
        supplied = ""
        if authorization:
            cleaned = authorization.strip()
            if cleaned.lower().startswith("bearer "):
                supplied = cleaned[7:].strip()
            else:
                supplied = cleaned
        if not supplied or not hmac.compare_digest(supplied, expected):
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
