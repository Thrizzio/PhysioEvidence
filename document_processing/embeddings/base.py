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

