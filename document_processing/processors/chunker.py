import re
import hashlib
from typing import List
from document_processing.models.document import StructuredDocument
from document_processing.models.chunk import Chunk, ChunkMetadata

class SectionAwareChunker:
    """
    Chunks a document while respecting section boundaries, then paragraphs, then sentences.
    """
    
    def __init__(self, target_chunk_size: int = 1000, chunk_overlap: int = 100, min_chunk_size: int = 50):
        # Sizes are currently in characters; can be migrated to tokens later
        self.target_chunk_size = target_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.strategy = "section-recursive-char-v1"
        self.version = "1"

    def _split_into_sentences(self, text: str) -> List[str]:
        """Naive sentence splitter."""
        # Split on period followed by space and capital letter.
        # This is basic and can be improved with nltk/spacy later.
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunks text hierarchically (paragraphs -> sentences -> hard split)
        """
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for p in paragraphs:
            if len(p) > self.target_chunk_size:
                # Paragraph is too big, split by sentences
                sentences = self._split_into_sentences(p)
                for s in sentences:
                    if current_length + len(s) > self.target_chunk_size and current_chunk:
                        chunks.append(" ".join(current_chunk))
                        # Basic overlap: keep last sentence
                        overlap_text = current_chunk[-1] if current_chunk else ""
                        current_chunk = [overlap_text, s] if overlap_text else [s]
                        current_length = sum(len(x) + 1 for x in current_chunk)
                    else:
                        current_chunk.append(s)
                        current_length += len(s) + 1
            else:
                if current_length + len(p) > self.target_chunk_size and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    # Overlap with the last paragraph
                    overlap_text = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_text, p] if overlap_text else [p]
                    current_length = sum(len(x) + 2 for x in current_chunk)
                else:
                    current_chunk.append(p)
                    current_length += len(p) + 2
                    
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
            
        return [c for c in chunks if len(c) >= self.min_chunk_size]

    def chunk(self, document: StructuredDocument) -> List[Chunk]:
        """
        Processes a StructuredDocument and returns a list of Chunks.
        """
        all_chunks = []
        chunk_index = 0

        for section_index, section in enumerate(document.sections):
            text_chunks = self._chunk_text(section.content)
            parent_id = hashlib.sha256(
                f"{document.document_id}:section:{section_index}:{section.name}".encode("utf-8")
            ).hexdigest()[:32]
            
            for text_chunk in text_chunks:
                stable_key = f"{document.document_id}:{self.strategy}:{self.version}:{chunk_index}:{text_chunk}"
                chunk_id = hashlib.sha256(stable_key.encode("utf-8")).hexdigest()
                metadata = ChunkMetadata(
                    document_title=document.metadata.title,
                    section=section.name,
                    page_start=section.page_start,
                    page_end=section.page_end,
                )
                
                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    content=text_chunk,
                    metadata=metadata,
                    parent_id=parent_id,
                    chunking_strategy=self.strategy,
                    chunking_version=self.version,
                    character_count=len(text_chunk),
                    token_count=len(text_chunk.split()),
                )
                
                all_chunks.append(chunk)
                chunk_index += 1
                
        return all_chunks
