"""Tests for Ingestion API application endpoints, CORS preflight, and authentication."""

import asyncio
import json
import pymupdf as fitz
import pytest

from document_processing.loaders.pdf_loader import PDFLoader
from document_processing.pipeline.document_pipeline import DocumentPipeline
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.processors.section_detector import SectionDetector
from services.ingestion_service.app import create_app
from services.ingestion_service.service import IngestionManager
from services.ingestion_service.storage import LocalOriginalStore
from storage.in_memory_store import InMemoryVectorStore


def make_valid_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (50, 50),
        "Abstract\nThis physiotherapy study contains enough extractable text for validation. "
        "The result is retained with a page reference for evidence inspection.",
    )
    content = doc.tobytes()
    doc.close()
    return content


def build_multipart_body(files: list[tuple[str, bytes, str]]) -> tuple[bytes, str]:
    boundary = "----WebKitFormBoundaryX9zB8y7w6v5u4t3s"
    lines = []
    for field_name, file_bytes, filename in files:
        lines.append(f"--{boundary}\r\n".encode("latin-1"))
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"\r\n'.encode("latin-1")
        )
        lines.append(b"Content-Type: application/pdf\r\n\r\n")
        lines.append(file_bytes)
        lines.append(b"\r\n")
    lines.append(f"--{boundary}--\r\n".encode("latin-1"))
    return b"".join(lines), f"multipart/form-data; boundary={boundary}"


async def asgi_request(app, method: str, path: str, headers: dict[str, str] | None = None, body: bytes = b""):
    raw_headers = []
    if headers:
        for k, v in headers.items():
            raw_headers.append((k.lower().encode("latin-1"), v.encode("latin-1")))

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": raw_headers,
    }
    response_started = {}
    response_body = bytearray()

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        if message["type"] == "http.response.start":
            response_started.update(message)
        elif message["type"] == "http.response.body":
            response_body.extend(message.get("body", b""))

    await app(scope, receive, send)

    res_headers = {
        k.decode("latin-1").lower(): v.decode("latin-1")
        for k, v in response_started.get("headers", [])
    }
    return response_started.get("status"), res_headers, bytes(response_body)


@pytest.fixture
def test_app(tmp_path, monkeypatch):
    monkeypatch.setenv("INGESTION_ADMIN_TOKEN", "valid_secret_token_123")
    store = InMemoryVectorStore()
    pipeline = DocumentPipeline(
        loader=PDFLoader(),
        section_detector=SectionDetector(),
        chunker=SectionAwareChunker(min_chunk_size=10),
        vector_store=store,
    )
    manager = IngestionManager(
        pipeline=pipeline,
        original_store=LocalOriginalStore(str(tmp_path / "originals")),
    )
    return create_app(manager=manager)


@pytest.mark.anyio
async def test_cors_preflight_success_127_0_0_1(test_app):
    status, headers, body = await asgi_request(
        test_app,
        method="OPTIONS",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization, content-type",
        },
    )
    assert status == 200
    assert headers.get("access-control-allow-origin") == "http://127.0.0.1:8081"
    assert "POST" in headers.get("access-control-allow-methods", "")
    allowed_headers = headers.get("access-control-allow-headers", "").lower()
    assert "authorization" in allowed_headers


@pytest.mark.anyio
async def test_cors_preflight_success_localhost(test_app):
    status, headers, body = await asgi_request(
        test_app,
        method="OPTIONS",
        path="/api/uploads",
        headers={
            "Origin": "http://localhost:8081",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    assert status == 200
    assert headers.get("access-control-allow-origin") == "http://localhost:8081"


@pytest.mark.anyio
async def test_cors_preflight_unauthorized_origin_rejected(test_app):
    status, headers, body = await asgi_request(
        test_app,
        method="OPTIONS",
        path="/api/uploads",
        headers={
            "Origin": "http://malicious-site.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization",
        },
    )
    # Origin is disallowed: either status 400 or missing allow-origin header
    assert status in (400, 403, 405) or "access-control-allow-origin" not in headers


@pytest.mark.anyio
async def test_upload_authenticated_success(test_app):
    pdf_bytes = make_valid_pdf()
    body, content_type = build_multipart_body([("files", pdf_bytes, "clinical_study.pdf")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Authorization": "Bearer valid_secret_token_123",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 200
    assert headers.get("access-control-allow-origin") == "http://127.0.0.1:8081"
    data = json.loads(response_body.decode("utf-8"))
    assert len(data) == 1
    assert data[0]["state"] == "succeeded"
    assert data[0]["filename"] == "clinical_study.pdf"


@pytest.mark.anyio
async def test_upload_raw_token_success(test_app):
    pdf_bytes = make_valid_pdf()
    body, content_type = build_multipart_body([("files", pdf_bytes, "clinical_study2.pdf")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Authorization": "valid_secret_token_123",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 200
    data = json.loads(response_body.decode("utf-8"))
    assert len(data) == 1
    assert data[0]["state"] == "succeeded"


@pytest.mark.anyio
async def test_upload_missing_token_rejected(test_app):
    pdf_bytes = make_valid_pdf()
    body, content_type = build_multipart_body([("files", pdf_bytes, "clinical_study.pdf")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 401
    data = json.loads(response_body.decode("utf-8"))
    assert "Admin authorization required" in data.get("detail", "")


@pytest.mark.anyio
async def test_upload_invalid_token_rejected(test_app):
    pdf_bytes = make_valid_pdf()
    body, content_type = build_multipart_body([("files", pdf_bytes, "clinical_study.pdf")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Authorization": "Bearer wrong_token_456",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 401
    data = json.loads(response_body.decode("utf-8"))
    assert "Admin authorization required" in data.get("detail", "")


@pytest.mark.anyio
async def test_upload_unset_admin_token_service_unavailable(test_app, monkeypatch):
    monkeypatch.delenv("INGESTION_ADMIN_TOKEN", raising=False)
    pdf_bytes = make_valid_pdf()
    body, content_type = build_multipart_body([("files", pdf_bytes, "clinical_study.pdf")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Authorization": "Bearer any_token",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 503
    data = json.loads(response_body.decode("utf-8"))
    assert "INGESTION_ADMIN_TOKEN is not configured" in data.get("detail", "")


@pytest.mark.anyio
async def test_upload_invalid_file_rejected(test_app):
    body, content_type = build_multipart_body([("files", b"not a pdf content", "notes.txt")])

    status, headers, response_body = await asgi_request(
        test_app,
        method="POST",
        path="/api/uploads",
        headers={
            "Origin": "http://127.0.0.1:8081",
            "Authorization": "Bearer valid_secret_token_123",
            "Content-Type": content_type,
        },
        body=body,
    )

    assert status == 200
    data = json.loads(response_body.decode("utf-8"))
    assert len(data) == 1
    assert data[0]["state"] == "failed"
    assert "PDF" in data[0]["error"]
