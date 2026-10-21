import pytest
from storage.in_memory_store import InMemoryVectorStore, cosine_similarity
from storage.base import FilterCriteria
from document_processing.models.document import StructuredDocument, DocumentMetadata, Section
from document_processing.models.chunk import Chunk, ChunkMetadata

def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0

    v4 = [1.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v4), 0.001) == 0.7071

def test_store_and_search_vector():
    store = InMemoryVectorStore()

    doc = StructuredDocument(
        document_id="doc_1",
        source_path="/path/paper1.pdf",
        filename="paper1.pdf",
        metadata=DocumentMetadata(
            title="ACL Rehabilitation Protocol",
            authors=["Smith J", "Doe A"],
            publication_year=2023,
            journal="Journal of Orthopaedic & Sports Physical Therapy"
        ),
        sections=[Section(name="Methods", content="Eccentric quadriceps training was conducted.")]
    )
    store.save_document(doc)

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            chunk_index=0,
            content="Eccentric quadriceps strengthening demonstrates superior hypertrophy post-ACL reconstruction.",
            metadata=ChunkMetadata(section="Methods", page_start=1, page_end=1),
            embedding=[1.0, 0.0, 0.0]
        ),
        Chunk(
            chunk_id="chunk_2",
            document_id="doc_1",
            chunk_index=1,
            content="Cryotherapy applied for 20 minutes reduces post-operative swelling and knee joint effusion.",
            metadata=ChunkMetadata(section="Results", page_start=2, page_end=2),
            embedding=[0.0, 1.0, 0.0]
        )
    ]
    store.save_chunks(chunks)

    assert store.count_documents() == 1
    assert store.count_chunks() == 2

    # Query vector close to chunk_1
    results = store.search_vector(query_embedding=[0.9, 0.1, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk_id == "chunk_1"
    assert results[0].document_title == "ACL Rehabilitation Protocol"
    assert results[0].score > results[1].score

def test_metadata_filtering():
    store = InMemoryVectorStore()

    doc1 = StructuredDocument(
        document_id="doc_old",
        source_path="/p1.pdf",
        filename="p1.pdf",
        metadata=DocumentMetadata(title="Old Study", publication_year=2015, authors=["Jones B"]),
        sections=[]
    )
    doc2 = StructuredDocument(
        document_id="doc_new",
        source_path="/p2.pdf",
        filename="p2.pdf",
        metadata=DocumentMetadata(title="Recent Study", publication_year=2024, authors=["Taylor C"]),
        sections=[]
    )
    store.save_document(doc1)
    store.save_document(doc2)

    store.save_chunks([
        Chunk(
            chunk_id="c1",
            document_id="doc_old",
            chunk_index=0,
            content="Isometric hamstring contractions at 45 degrees flexion.",
            metadata=ChunkMetadata(section="Methods"),
            embedding=[0.5, 0.5, 0.0]
        ),
        Chunk(
            chunk_id="c2",
            document_id="doc_new",
            chunk_index=0,
            content="Isometric hamstring contractions at 60 degrees flexion.",
            metadata=ChunkMetadata(section="Methods"),
            embedding=[0.5, 0.5, 0.0]
        ),
        Chunk(
            chunk_id="c3",
            document_id="doc_new",
            chunk_index=1,
            content="Discussion on patient adherence to home exercises.",
            metadata=ChunkMetadata(section="Discussion"),
            embedding=[0.5, 0.5, 0.0]
        )
    ])

    # Filter by section
    results = store.search_vector(
        query_embedding=[0.5, 0.5, 0.0],
        filter_criteria=FilterCriteria(sections=["Discussion"])
    )
    assert len(results) == 1
    assert results[0].chunk_id == "c3"

    # Filter by min_year
    results_year = store.search_vector(
        query_embedding=[0.5, 0.5, 0.0],
        filter_criteria=FilterCriteria(min_year=2020)
    )
    assert len(results_year) == 2
    assert all(r.document_id == "doc_new" for r in results_year)

    # Filter by author
    results_author = store.search_vector(
        query_embedding=[0.5, 0.5, 0.0],
        filter_criteria=FilterCriteria(authors_contain="Taylor")
    )
    assert len(results_author) == 2
    assert all(r.document_id == "doc_new" for r in results_author)

def test_search_text():
    store = InMemoryVectorStore()
    doc = StructuredDocument(
        document_id="doc_patella",
        source_path="/p.pdf",
        filename="p.pdf",
        metadata=DocumentMetadata(title="Patellofemoral Pain Syndrome"),
        sections=[]
    )
    store.save_document(doc)
    store.save_chunks([
        Chunk(
            chunk_id="c_pain",
            document_id="doc_patella",
            chunk_index=0,
            content="Patellofemoral pain syndrome is exacerbated by stair ambulation and deep squatting.",
            metadata=ChunkMetadata(section="Introduction")
        ),
        Chunk(
            chunk_id="c_other",
            document_id="doc_patella",
            chunk_index=1,
            content="Cardiovascular fitness training improves general endurance.",
            metadata=ChunkMetadata(section="Discussion")
        )
    ])

    results = store.search_text("patellofemoral pain squatting", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk_id == "c_pain"

def test_delete_document_cascades():
    store = InMemoryVectorStore()
    doc = StructuredDocument(
        document_id="doc_to_delete",
        source_path="/del.pdf",
        filename="del.pdf",
        metadata=DocumentMetadata(title="Temporary Paper"),
        sections=[]
    )
    store.save_document(doc)
    store.save_chunks([
        Chunk(chunk_id="chunk_del_1", document_id="doc_to_delete", chunk_index=0, content="Some text"),
        Chunk(chunk_id="chunk_del_2", document_id="doc_to_delete", chunk_index=1, content="More text")
    ])

    assert store.count_documents() == 1
    assert store.count_chunks() == 2

    deleted = store.delete_document("doc_to_delete")
    assert deleted is True
    assert store.count_documents() == 0
    assert store.count_chunks() == 0
    assert store.get_chunk("chunk_del_1") is None


def test_has_document():
    store = InMemoryVectorStore()
    assert store.has_document("nonexistent") is False

    doc = StructuredDocument(
        document_id="doc_xyz",
        source_path="/path/test.pdf",
        filename="test.pdf",
        metadata=DocumentMetadata(title="Test", content_hash="hash_123456"),
        sections=[]
    )
    store.save_document(doc)

    assert store.has_document("doc_xyz") is True
    assert store.has_document("hash_123456") is True
    assert store.has_document("other_hash") is False

    store.delete_document("doc_xyz")
    assert store.has_document("doc_xyz") is False
    assert store.has_document("hash_123456") is False

