from abc import ABC, abstractmethod
from typing import List
from document_processing.models.chunk import Chunk

class EmbeddingProvider(ABC):
    """
    Base abstraction for embedding providers.
    """
    
    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Takes a list of chunks, generates embeddings for them, 
        and returns the chunks with their embeddings attached.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a search query.
        """
        pass

    @property
    def dimensions(self) -> int:
        """Output dimensionality used to validate index compatibility."""
        raise NotImplementedError

    @property
    def model_id(self) -> str:
        """Stable model/version identifier stored with processing metadata."""
        raise NotImplementedError

    @property
    def normalized(self) -> bool:
        """Whether output vectors are L2-normalized."""
        return False
