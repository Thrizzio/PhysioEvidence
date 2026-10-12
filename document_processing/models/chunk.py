from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    """
    Preserves context about where a chunk came from.
    """
    document_title: Optional[str] = None
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # Flexible field for additional context (e.g. authors, year)
    extra: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    """
    Represents a chunk of text ready for embedding.
    """
    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)

    # Re-indexing and evaluation metadata. The defaults preserve compatibility
    # with existing callers that construct Chunk directly.
    parent_id: Optional[str] = None
    chunking_strategy: str = "section-recursive-char-v1"
    chunking_version: str = "1"
    character_count: int = 0
    token_count: int = 0
    
    # Optional embedding if attached later
    embedding: Optional[List[float]] = None
