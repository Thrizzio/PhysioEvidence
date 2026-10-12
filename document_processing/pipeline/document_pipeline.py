import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from document_processing.loaders.base import DocumentLoader
from document_processing.processors.section_detector import SectionDetector
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.embeddings.base import EmbeddingProvider
from document_processing.models.document import StructuredDocument
from document_processing.models.chunk import Chunk
from storage.base import VectorStore
from observability.tracing import trace_operation

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
    Coordinates loading, text cleaning, section detection, chunking,
    optional embedding generation, and persistent vector storage.
    """
    
    def __init__(
        self,
        loader: DocumentLoader,
        section_detector: SectionDetector,
        chunker: SectionAwareChunker,
        embedder: Optional[EmbeddingProvider] = None,
        vector_store: Optional[VectorStore] = None
    ):
        self.loader = loader
        self.section_detector = section_detector
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        
    @trace_operation(name="document_pipeline_process", span_type="chain")
    def process(self, file_path: str, generate_embeddings: bool = True) -> List[Chunk]:
        """
        Runs a single document through the complete pipeline.
        Optionally generates embeddings and persists to vector store.
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
            
        # 2. Clean & Detect Sections
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
            if not chunks:
                raise ValueError("No indexable text chunks were produced.")
        except Exception as e:
            raise DocumentProcessingError(file_path, "chunking", str(e)) from e

        # 4. Embed
        if generate_embeddings and self.embedder is None:
            raise DocumentProcessingError(
                file_path,
                "embedding",
                "Embedding generation was requested but no embedding provider is configured.",
            )
        if generate_embeddings and self.embedder:
            try:
                chunks = self.embedder.embed_chunks(chunks)
                if self.vector_store is not None:
                    # Persist the compatibility contract beside each vector.
                    setattr(self.vector_store, "embedding_model", self.embedder.model_id)
                    setattr(self.vector_store, "embedding_dimensions", self.embedder.dimensions)
                    setattr(self.vector_store, "embedding_normalized", self.embedder.normalized)
                logger.info(f"Generated embeddings for {len(chunks)} chunks")
            except Exception as e:
                raise DocumentProcessingError(file_path, "embedding", str(e)) from e
                
        # 5. Persist to Vector Store
        if self.vector_store is not None:
            try:
                self.vector_store.save_document(structured_doc)
                self.vector_store.save_chunks(chunks)
                logger.info(f"Saved document and {len(chunks)} chunks to vector store")
            except Exception as e:
                raise DocumentProcessingError(file_path, "storage", str(e)) from e

        duration = time.time() - start_time
        logger.info(f"Processing completed in {duration:.1f} seconds")
        
        return chunks

    def process_directory(
        self,
        directory_path: str,
        generate_embeddings: bool = True,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Batch-processes all PDF files in a given directory.
        Tolerates single-file failures and reports a complete execution summary.
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"Directory does not exist or is not a directory: {directory_path}")

        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(list(path.glob(pattern)))
        
        start_time = time.time()
        logger.info(f"Starting batch ingestion of {len(pdf_files)} PDF files in {directory_path}")

        successful: List[str] = []
        failed: List[Dict[str, str]] = []
        total_chunks = 0

        for pdf_path in pdf_files:
            file_str = str(pdf_path)
            try:
                chunks = self.process(file_str, generate_embeddings=generate_embeddings)
                successful.append(file_str)
                total_chunks += len(chunks)
            except Exception as e:
                logger.error(f"Failed processing {file_str}: {e}")
                failed.append({"file": file_str, "error": str(e)})

        total_duration = time.time() - start_time
        logger.info(
            f"Batch ingestion finished: {len(successful)} succeeded, {len(failed)} failed, "
            f"{total_chunks} total chunks in {total_duration:.1f}s"
        )

        return {
            "total_files": len(pdf_files),
            "successful_files": successful,
            "failed_files": failed,
            "total_chunks": total_chunks,
            "duration_seconds": round(total_duration, 2)
        }
