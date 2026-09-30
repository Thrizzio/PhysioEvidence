from unittest.mock import MagicMock
import pytest
from storage.in_memory_store import InMemoryVectorStore
from storage.base import FilterCriteria
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata
from retrieval.vector_retriever import VectorRetriever

@pytest.fixture
def populated_store():
    store = InMemoryVectorStore()
    doc = StructuredDocument(
        document_id="doc_acl",
        source_path="/acl.pdf",
        filename="acl.pdf",
        metadata=DocumentMetadata(title="ACL Rehabilitation Study", publication_year=2023),
        sections=[]
    )
    store.save_document(doc)

    chunks = [
        Chunk(
            chunk_id="c_quad",
            document_id="doc_acl",
            chunk_index=0,
            content="Quadriceps index greater than 90% is a key criterion for return to sport.",
            metadata=ChunkMetadata(section="Results"),
            embedding=[1.0, 0.0, 0.0]
        ),
        Chunk(
            chunk_id="c_ham",
            document_id="doc_acl",
            chunk_index=1,
            content="Hamstring autograft demonstrates gradual revascularization over 12 months.",
            metadata=ChunkMetadata(section="Discussion"),
            embedding=[0.0, 1.0, 0.0]
        )
    ]
    store.save_chunks(chunks)
    return store

def test_vector_retriever_basic(populated_store):
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.95, 0.05, 0.0]

    retriever = VectorRetriever(vector_store=populated_store, embedder=mock_embedder)
    results = retriever.retrieve("quadriceps strength return to sport", top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "c_quad"
    assert results[0].document_title == "ACL Rehabilitation Study"
    mock_embedder.embed_query.assert_called_once_with("quadriceps strength return to sport")

def test_vector_retriever_with_filters(populated_store):
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.5, 0.5, 0.0]

    retriever = VectorRetriever(vector_store=populated_store, embedder=mock_embedder)
    results = retriever.retrieve(
        "tissue healing",
        top_k=5,
        filter_criteria=FilterCriteria(sections=["Discussion"])
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c_ham"

def test_vector_retriever_empty_query(populated_store):
    mock_embedder = MagicMock()
    retriever = VectorRetriever(vector_store=populated_store, embedder=mock_embedder)
    results = retriever.retrieve("   ")
    assert results == []
    assert not mock_embedder.embed_query.called
