import os
import time
import logging
from typing import List
import google.generativeai as genai
from document_processing.embeddings.base import EmbeddingProvider
from document_processing.models.chunk import Chunk

logger = logging.getLogger(__name__)

class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Generates embeddings using Google's Gemini API.
    """
    
    def __init__(self, model_name: str = "models/text-embedding-004", batch_size: int = 100):
        self.model_name = model_name
        self.batch_size = batch_size
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
            
        genai.configure(api_key=api_key)

    @property
    def dimensions(self) -> int:
        return 768

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def normalized(self) -> bool:
        # Gemini's API behavior is provider-defined; cosine distance does not
        # require us to claim normalization here.
        return False

    def embed_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Generates embeddings for chunks in batches.
        """
        # Filter chunks that already have embeddings
        chunks_to_embed = [c for c in chunks if c.embedding is None]
        
        if not chunks_to_embed:
            return chunks

        total_batches = (len(chunks_to_embed) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(chunks_to_embed), self.batch_size):
            batch = chunks_to_embed[i:i + self.batch_size]
            texts = [c.content for c in batch]
            
            try:
                logger.debug(f"Generating embeddings for batch {i//self.batch_size + 1}/{total_batches}")
                # For Gemini, embed_content can take a list of strings
                result = genai.embed_content(
                    model=self.model_name,
                    content=texts,
                    task_type="retrieval_document"
                )
                
                embeddings = result['embedding']
                
                # Attach to chunks
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    
                # Simple rate limit prevention
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {str(e)}")
                raise RuntimeError(f"Embedding API failure: {str(e)}") from e
                
        return chunks

    def embed_query(self, query: str) -> List[float]:
        """
        Generates an embedding vector for a single query using task_type='retrieval_query'.
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise RuntimeError(f"Query embedding API failure: {str(e)}") from e
