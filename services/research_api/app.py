"""Public API for the minimal research chat website."""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generation.synthesizer import EvidenceAnswer
from services.common import build_rag, configured_origins
from storage.base import FilterCriteria

logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    sections: Optional[list[str]] = None
    min_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    max_year: Optional[int] = Field(default=None, ge=1800, le=2100)
    hybrid: bool = True


class QueryResponse(BaseModel):
    answer: EvidenceAnswer
    retrieval_strategy: str


def create_app(rag=None) -> FastAPI:
    app = FastAPI(title="PhysioEvidence Research API", version="1.0")
    origins = configured_origins("RESEARCH_UI_ORIGIN")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
    app.state.rag = rag

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "research-api", "ready": app.state.rag is not None}

    @app.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        if app.state.rag is None:
            try:
                app.state.rag = build_rag(hybrid=request.hybrid)
            except Exception as exc:
                logger.exception("Research API dependency initialization failed")
                raise HTTPException(status_code=503, detail="Research service is not configured.") from exc

        if request.min_year is not None and request.max_year is not None and request.min_year > request.max_year:
            raise HTTPException(status_code=422, detail="min_year cannot be greater than max_year.")

        filters = FilterCriteria(
            sections=request.sections,
            min_year=request.min_year,
            max_year=request.max_year,
        )
        try:
            answer = app.state.rag.ask(
                request.question,
                top_k=request.top_k,
                filter_criteria=filters,
            )
        except Exception as exc:
            logger.exception("Research query failed")
            raise HTTPException(status_code=502, detail="The research query could not be completed.") from exc
        return QueryResponse(
            answer=answer,
            retrieval_strategy="hybrid-rrf" if request.hybrid else "dense-vector",
        )

    return app


app = create_app()
