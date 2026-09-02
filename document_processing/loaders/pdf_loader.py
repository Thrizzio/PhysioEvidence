import os
import uuid
import fitz  # PyMuPDF
from document_processing.loaders.base import DocumentLoader
from document_processing.models.document import ExtractedDocument, Page, DocumentMetadata

class PDFLoader(DocumentLoader):
    """
    Loader for PDF documents using PyMuPDF.
    """
    
    def load(self, file_path: str) -> ExtractedDocument:
        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"File is not a PDF: {file_path}")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        document_id = str(uuid.uuid4())
        filename = os.path.basename(file_path)
        
        pages = []
        try:
            with fitz.open(file_path) as doc:
                metadata_dict = doc.metadata or {}
                # Basic metadata extraction
                doc_metadata = DocumentMetadata(
                    title=metadata_dict.get("title"),
                    authors=[metadata_dict.get("author")] if metadata_dict.get("author") else None,
                )
                
                total_text_length = 0
                for page_num, page in enumerate(doc):
                    text = page.get_text("text")
                    # Preserve page boundaries (1-indexed for humans)
                    pages.append(Page(page_number=page_num + 1, text=text))
                    total_text_length += len(text.strip())
                    
                # If there are pages but almost no text, it might need OCR
                if len(doc) > 0 and (total_text_length / len(doc)) < 50:
                    # Arbitrary threshold: if avg chars per page is < 50, probably a scanned document
                    raise ValueError(f"PDF contains very little extractable text. OCR may be required. (Avg chars/page: {total_text_length / len(doc):.1f})")
                    
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise RuntimeError(f"Failed to process PDF {file_path}: {str(e)}") from e
            
        return ExtractedDocument(
            document_id=document_id,
            source_path=file_path,
            filename=filename,
            metadata=doc_metadata,
            pages=pages
        )
