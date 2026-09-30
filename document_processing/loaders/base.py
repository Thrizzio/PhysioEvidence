from abc import ABC, abstractmethod
from document_processing.models.document import ExtractedDocument

class DocumentLoader(ABC):
    """
    Base abstraction for all document loaders.
    """
    
    @abstractmethod
    def load(self, file_path: str) -> ExtractedDocument:
        """
        Load a document from the given file path and return an ExtractedDocument.
        """
        pass

