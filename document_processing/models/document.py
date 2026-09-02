from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_year: Optional[int] = None
    journal: Optional[str] = None
    doi: Optional[str] = Field(None, alias="DOI")
    source_url: Optional[str] = None
    # Additional flexible metadata
    extra: Dict[str, Any] = Field(default_factory=dict)

class Page(BaseModel):
    page_number: int
    text: str

class ExtractedDocument(BaseModel):
    """
    Represents a document after initial extraction but before cleaning or chunking.
    Preserves page boundaries as requested.
    """
    document_id: str
    source_path: str
    filename: str
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    pages: List[Page] = Field(default_factory=list)

    @property
    def raw_text(self) -> str:
        """Helper to get the full raw text."""
        return "\n\n".join(page.text for page in self.pages)

class Section(BaseModel):
    """
    Represents a detected section within the document.
    """
    name: str
    content: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None

class StructuredDocument(BaseModel):
    """
    Represents a document after cleaning and section detection.
    """
    document_id: str
    source_path: str
    filename: str
    metadata: DocumentMetadata
    sections: List[Section] = Field(default_factory=list)

