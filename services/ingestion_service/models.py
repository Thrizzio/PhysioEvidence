from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field


class JobState(StrEnum):
    pending = "pending"
    processing = "processing"
    succeeded = "succeeded"
    duplicate = "duplicate"
    failed = "failed"


class IngestionJob(BaseModel):
    job_id: str
    filename: str
    content_hash: str
    state: JobState = JobState.pending
    progress: int = Field(default=0, ge=0, le=100)
    document_id: Optional[str] = None
    chunk_count: int = 0
    file_size_bytes: int = 0
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class StorageUsage(BaseModel):
    file_count: int
    bytes_used: int
    failed_uploads: int
    quota_bytes: Optional[int] = None
    warning_threshold: float
    estimated_remaining_bytes: Optional[int] = None


class DocumentStatus(BaseModel):
    document_id: Optional[str]
    filename: str
    content_hash: str
    state: JobState
    chunk_count: int
    file_size_bytes: int
    error: Optional[str] = None
