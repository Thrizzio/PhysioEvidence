import pytest
from unittest.mock import MagicMock
from storage.base import SearchResult
from storage.in_memory_store import InMemoryVectorStore
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata
from retrieval.hybrid_retriever import HybridRetriever, reciprocal_rank_fusion

def test_reciprocal_rank_fusion_math():
    dense_results = [
        SearchResult(chunk_id="chunk_A", document_id="doc_1", chunk_index=0, content="Text A", score=0.9),
        SearchResult(chunk_id="chunk_B", document_id="doc_1", chunk_index=1, content="Text B", score=0.7),
    ]

    sparse_results = [
        SearchResult(chunk_id="chunk_B", document_id="doc_1", chunk_index=1, content="Text B", score=5.2),
        SearchResult(chunk_id="chunk_C", document_id="doc_2", chunk_index=0, content="Text C", score=3.1),
    ]

    # k=60:
    # chunk_A: 1/(60+1) = 1/61 = 0.016393
    # chunk_B: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.016129 + 0.016393 = 0.032522
    # chunk_C: 1/(60+2) = 1/62 = 0.016129
    # Chunk B should rank #1 because it has evidence in BOTH channels!
    fused = reciprocal_rank_fusion(dense_results, sparse_results, k=60, top_k=3)

    assert len(fused) == 3
    assert fused[0].chunk_id == "chunk_B"
    assert fused[1].chunk_id == "chunk_A"
    assert fused[2].chunk_id == "chunk_C"
    assert fused[0].score > fused[1].score

def test_hybrid_retriever_execution():
    store = InMemoryVectorStore()
    doc = StructuredDocument(
        document_id="doc_shoulder",
        source_path="/s.pdf",
        filename="s.pdf",
        metadata=DocumentMetadata(title="Rotator Cuff Rehabilitation"),
        sections=[]
    )
    store.save_document(doc)

    store.save_chunks([
        Chunk(
            chunk_id="c_supraspinatus",
            document_id="doc_shoulder",
            chunk_index=0,
            content="Supraspinatus tendon tears respond well to progressive resisted external rotation exercises.",
            metadata=ChunkMetadata(section="Results"),
            embedding=[0.8, 0.2, 0.0]
        ),
        Chunk(
            chunk_id="c_scapular",
            document_id="doc_shoulder",
            chunk_index=1,
            content="Scapular dyskinesis alters glenohumeral biomechanics during elevation.",
            metadata=ChunkMetadata(section="Discussion"),
            embedding=[0.2, 0.8, 0.0]
        )
    ])

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.8, 0.2, 0.0]

    retriever = HybridRetriever(
        vector_store=store,
        embedder=mock_embedder,
        dense_weight=1.0,
        sparse_weight=1.0
    )

    results = retriever.retrieve("supraspinatus tendon external rotation", top_k=2)

    assert len(results) >= 1
    assert results[0].chunk_id == "c_supraspinatus"
    assert results[0].document_title == "Rotator Cuff Rehabilitation"
