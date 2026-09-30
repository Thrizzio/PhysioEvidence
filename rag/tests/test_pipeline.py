import pytest
from storage.in_memory_store import InMemoryVectorStore
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata
from retrieval.vector_retriever import VectorRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.llm import MockLLMProvider
from generation.synthesizer import EvidenceSynthesizer
from rag.pipeline import PhysioEvidenceRAG
from storage.base import FilterCriteria

@pytest.fixture
def rag_system():
    store = InMemoryVectorStore()
    doc = StructuredDocument(
        document_id="doc_achilles",
        source_path="/achilles.pdf",
        filename="achilles.pdf",
        metadata=DocumentMetadata(title="Achilles Tendon Loading", publication_year=2022),
        sections=[]
    )
    store.save_document(doc)

    store.save_chunks([
        Chunk(
            chunk_id="chunk_hsr",
            document_id="doc_achilles",
            chunk_index=0,
            content="Heavy slow resistance training yielded equal clinical efficacy to eccentric training at 12 weeks.",
            metadata=ChunkMetadata(section="Results", page_start=5, page_end=6),
            embedding=[0.9, 0.1, 0.0]
        ),
        Chunk(
            chunk_id="chunk_alfredson",
            document_id="doc_achilles",
            chunk_index=1,
            content="The Alfredson eccentric protocol requires 180 repetitions per day.",
            metadata=ChunkMetadata(section="Methods", page_start=2, page_end=3),
            embedding=[0.7, 0.3, 0.0]
        )
    ])

    retriever = VectorRetriever(vector_store=store, embedder=lambda q: [0.9, 0.1, 0.0])
    mock_llm = MockLLMProvider(
        response_generator=lambda p: "Heavy slow resistance is as effective as eccentric loading for Achilles tendinopathy [Source 1]."
    )
    synthesizer = EvidenceSynthesizer(llm=mock_llm)
    reranker = CrossEncoderReranker()

    rag = PhysioEvidenceRAG(retriever=retriever, synthesizer=synthesizer, reranker=reranker)
    return rag

def test_rag_ask_end_to_end(rag_system):
    answer = rag_system.ask("What is the evidence for heavy slow resistance in Achilles tendinopathy?", top_k=2)

    assert answer.insufficient_evidence is False
    assert "[Source 1]" in answer.answer
    assert len(answer.sources) == 2
    assert answer.sources[0].title == "Achilles Tendon Loading"
    assert answer.latency_ms > 0

def test_rag_ask_with_filter(rag_system):
    filters = FilterCriteria(sections=["Methods"])
    answer = rag_system.ask("Tell me about protocols", top_k=2, filter_criteria=filters)

    # Only Methods section chunk should be retrieved
    assert len(answer.sources) == 1
    assert answer.sources[0].section == "Methods"
