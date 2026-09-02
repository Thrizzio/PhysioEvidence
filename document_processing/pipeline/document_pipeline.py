import logging
import time
from typing import List, Optional
from document_processing.loaders.base import DocumentLoader
from document_processing.processors.section_detector import SectionDetector
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.embeddings.base import EmbeddingProvider
from document_processing.models.document import StructuredDocument
from document_processing.models.chunk import Chunk

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    def __init__(self, document: str, stage: str, reason: str):
        self.document = document
        self.stage = stage
        self.reason = reason
        super().__init__(f"DocumentProcessingError: document='{document}', stage='{stage}', reason='{reason}'")

class DocumentPipeline:
    """
    Central orchestration layer for the document processing pipeline.
    """
    
    def __init__(
        self,
        loader: DocumentLoader,
        section_detector: SectionDetector,
        chunker: SectionAwareChunker,
        embedder: Optional[EmbeddingProvider] = None
    ):
        self.loader = loader
        self.section_detector = section_detector
        self.chunker = chunker
        self.embedder = embedder
        
    def process(self, file_path: str, generate_embeddings: bool = True) -> List[Chunk]:
        """
        Runs the document through the complete pipeline.
        """
        start_time = time.time()
        logger.info(f"Processing document: {file_path}")
        
        # 1. Load & Extract
        try:
            extracted_doc = self.loader.load(file_path)
            logger.info(f"Extracted {len(extracted_doc.pages)} pages")
            logger.info(f"Extracted {len(extracted_doc.raw_text)} characters")
        except Exception as e:
            raise DocumentProcessingError(file_path, "extraction", str(e)) from e
            
        # 2. Clean & Detect Sections (Cleaning happens inside section detector per requirements)
        try:
            sections = self.section_detector.detect_sections(extracted_doc)
            
            structured_doc = StructuredDocument(
                document_id=extracted_doc.document_id,
                source_path=extracted_doc.source_path,
                filename=extracted_doc.filename,
                metadata=extracted_doc.metadata,
                sections=sections
            )
            logger.info(f"Detected {len(sections)} sections")
        except Exception as e:
            raise DocumentProcessingError(file_path, "structuring", str(e)) from e
            
        # 3. Chunk
        try:
            chunks = self.chunker.chunk(structured_doc)
            logger.info(f"Generated {len(chunks)} chunks")
        except Exception as e:
            raise DocumentProcessingError(file_path, "chunking", str(e)) from e
            
        # 4. Embed
        if generate_embeddings and self.embedder:
            try:
                chunks = self.embedder.embed_chunks(chunks)
                logger.info(f"Generated embeddings for {len(chunks)} chunks")
            except Exception as e:
                raise DocumentProcessingError(file_path, "embedding", str(e)) from e
                
        duration = time.time() - start_time
        logger.info(f"Processing completed in {duration:.1f} seconds")
        
        return chunks

