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
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type", "Accept", "Origin", "X-Requested-With"],
            allow_credentials=True,
        )
    app.state.rags = {"hybrid": rag} if rag is not None else {}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "service": "research-api", "ready": bool(app.state.rags)}

    @app.post("/api/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        strategy_key = "hybrid" if request.hybrid else "dense"
        if strategy_key not in app.state.rags:
            try:
                app.state.rags[strategy_key] = build_rag(hybrid=request.hybrid)
            except Exception as exc:
                logger.exception("Research API dependency initialization failed")
                raise HTTPException(status_code=503, detail=f"Research service is not configured: {exc}") from exc

        if request.min_year is not None and request.max_year is not None and request.min_year > request.max_year:
            raise HTTPException(status_code=422, detail="min_year cannot be greater than max_year.")

        filters = FilterCriteria(
            sections=request.sections,
            min_year=request.min_year,
            max_year=request.max_year,
        )
        try:
            answer = app.state.rags[strategy_key].ask(
                request.question,
                top_k=request.top_k,
                filter_criteria=filters,
            )
        except Exception as exc:
            logger.exception("Research query failed")
            raise HTTPException(status_code=502, detail=f"The research query could not be completed: {exc}") from exc

        return QueryResponse(
            answer=answer,
            retrieval_strategy="hybrid-rrf" if request.hybrid else "dense-vector",
        )

    return app


app = create_app()
