from typing import List, Optional
from pydantic import BaseModel, Field
from storage.base import SearchResult

class CitationSource(BaseModel):
    """
    Metadata representation of an evidence citation source.
    """
    source_number: int
    chunk_id: str
    document_id: str
    title: Optional[str] = None
    section: Optional[str] = None
    pages: Optional[str] = None
    similarity_score: float

class AssembledContext(BaseModel):
    """
    The formatted context string and mapping of numbered sources.
    """
    context_text: str
    sources: List[CitationSource] = Field(default_factory=list)
    total_chunks: int

class ContextAssembler:
    """
    Formats retrieved search results into structured, numbered evidence blocks.
    Enforces length budgets to protect LLM context windows.
    """

    def __init__(self, max_characters: int = 12000):
        self.max_characters = max_characters

    def assemble(self, results: List[SearchResult]) -> AssembledContext:
        if not results:
            return AssembledContext(
                context_text="No relevant research literature was retrieved for this inquiry.",
                sources=[],
                total_chunks=0
            )

        blocks: List[str] = []
        sources: List[CitationSource] = []
        accumulated_chars = 0

        for idx, item in enumerate(results, start=1):
            title = item.document_title or f"Document {item.document_id}"
            section = item.section or "Unspecified Section"
            
            # Format page range
            if item.page_start and item.page_end:
                pages_str = f"pp. {item.page_start}-{item.page_end}" if item.page_start != item.page_end else f"p. {item.page_start}"
            elif item.page_start:
                pages_str = f"p. {item.page_start}"
            else:
                pages_str = "p. N/A"

            header = f"[Source {idx}] Title: {title} | Section: {section} ({pages_str})"
            block = f"{header}\n{item.content.strip()}"

            if accumulated_chars + len(block) > self.max_characters and sources:
                # Stop if exceeding character budget
                break

            blocks.append(block)
            sources.append(CitationSource(
                source_number=idx,
                chunk_id=item.chunk_id,
                document_id=item.document_id,
                title=title,
                section=section,
                pages=pages_str,
                similarity_score=round(item.score, 4)
            ))
            accumulated_chars += len(block) + 2

        formatted_context = "\n\n---\n\n".join(blocks)
        return AssembledContext(
            context_text=formatted_context,
            sources=sources,
            total_chunks=len(sources)
        )
