import pytest
from document_processing.processors.chunker import SectionAwareChunker
from document_processing.models.document import StructuredDocument, Section, DocumentMetadata

@pytest.fixture
def structured_doc():
    return StructuredDocument(
        document_id="doc1",
        source_path="test.pdf",
        filename="test.pdf",
        metadata=DocumentMetadata(title="Test Doc"),
        sections=[
            Section(name="Abstract", content="Short abstract."),
            Section(name="Introduction", content="This is paragraph one.\n\nThis is paragraph two. It has multiple sentences. We need to test if it splits properly.")
        ]
    )

def test_chunker_basic(structured_doc):
    chunker = SectionAwareChunker(target_chunk_size=100, min_chunk_size=10)
    chunks = chunker.chunk(structured_doc)
    
    assert len(chunks) > 0
    assert chunks[0].metadata.section == "Abstract"
    assert chunks[0].metadata.document_title == "Test Doc"
    assert chunks[0].document_id == "doc1"
    
def test_chunker_hierarchy(structured_doc):
    # Set chunk size very small to force sentence splitting
    chunker = SectionAwareChunker(target_chunk_size=30, min_chunk_size=5)
    chunks = chunker.chunk(structured_doc)
    
    # "This is paragraph two. It has multiple sentences. We need to test if it splits properly." 
    # should be split into multiple chunks
    intro_chunks = [c for c in chunks if c.metadata.section == "Introduction"]
    assert len(intro_chunks) >= 3
