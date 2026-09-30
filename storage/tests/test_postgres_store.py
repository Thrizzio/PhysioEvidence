from unittest.mock import MagicMock
import pytest
from storage.postgres_store import PostgresVectorStore
from storage.base import FilterCriteria
from document_processing.models.document import StructuredDocument, DocumentMetadata
from document_processing.models.chunk import Chunk, ChunkMetadata

@pytest.fixture
def mock_store():
    store = PostgresVectorStore(connection_string="postgresql://mock:mock@localhost:5432/mock_db")
    return store

def _setup_mock_connection(mock_store):
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_cur = MagicMock()
    mock_cur.__enter__.return_value = mock_cur
    mock_conn.cursor.return_value = mock_cur
    mock_store._get_conn = MagicMock(return_value=mock_conn)
    return mock_conn, mock_cur

def test_save_document_sql(mock_store):
    mock_conn, mock_cur = _setup_mock_connection(mock_store)

    doc = StructuredDocument(
        document_id="doc_123",
        source_path="/path/test.pdf",
        filename="test.pdf",
        metadata=DocumentMetadata(
            title="Stroke Rehabilitation",
            authors=["Alice Wonder", "Bob Builder"],
            publication_year=2024,
            journal="Neurorehabilitation"
        ),
        sections=[]
    )

    mock_store.save_document(doc)

    assert mock_cur.execute.called
    sql_call = mock_cur.execute.call_args[0][0]
    params_call = mock_cur.execute.call_args[0][1]

    assert "INSERT INTO documents" in sql_call
    assert params_call["doc_id"] == "doc_123"
    assert params_call["title"] == "Stroke Rehabilitation"
    assert params_call["year"] == 2024

def test_save_chunks_sql(mock_store):
    mock_conn, mock_cur = _setup_mock_connection(mock_store)

    chunks = [
        Chunk(
            chunk_id="chunk_1",
            document_id="doc_123",
            chunk_index=0,
            content="Task-oriented arm training improves upper limb function.",
            metadata=ChunkMetadata(section="Results", page_start=3, page_end=3),
            embedding=[0.1] * 768
        )
    ]

    mock_store.save_chunks(chunks)

    assert mock_cur.executemany.called
    sql_call = mock_cur.executemany.call_args[0][0]
    rows_call = mock_cur.executemany.call_args[0][1]

    assert "INSERT INTO chunks" in sql_call
    assert len(rows_call) == 1
    assert rows_call[0]["chunk_id"] == "chunk_1"
    assert rows_call[0]["section"] == "Results"
    assert rows_call[0]["embedding"] is not None

def test_search_vector_sql_building(mock_store):
    mock_conn, mock_cur = _setup_mock_connection(mock_store)
    mock_cur.fetchall.return_value = [
        {
            "chunk_id": "c1",
            "document_id": "doc_1",
            "chunk_index": 0,
            "content": "Sample content",
            "section": "Methods",
            "page_start": 1,
            "page_end": 1,
            "extra_metadata": {},
            "document_title": "Paper 1",
            "score": 0.88
        }
    ]

    filter_criteria = FilterCriteria(
        document_ids=["doc_1"],
        sections=["Methods"],
        min_year=2020,
        max_year=2024,
        authors_contain="Smith"
    )

    results = mock_store.search_vector(
        query_embedding=[0.0] * 768,
        top_k=3,
        filter_criteria=filter_criteria
    )

    assert mock_cur.execute.called
    sql_call = mock_cur.execute.call_args[0][0]
    assert "c.document_id = ANY(%(doc_ids)s)" in sql_call
    assert "c.section = ANY(%(sections)s)" in sql_call
    assert "d.publication_year >= %(min_year)s" in sql_call
    assert "d.publication_year <= %(max_year)s" in sql_call
    assert "array_to_string(d.authors, ' ') ILIKE %(author)s" in sql_call
    assert "ORDER BY c.embedding <=> %(query_emb)s ASC" in sql_call
    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].score == 0.88

def test_search_text_sql_building(mock_store):
    mock_conn, mock_cur = _setup_mock_connection(mock_store)
    mock_cur.fetchall.return_value = []

    mock_store.search_text(
        query_text="rotator cuff tendinopathy",
        top_k=5,
        filter_criteria=FilterCriteria(sections=["Results"])
    )

    assert mock_cur.execute.called
    sql_call = mock_cur.execute.call_args[0][0]
    assert "c.tsv @@ plainto_tsquery('english', %(query_text)s)" in sql_call
    assert "c.section = ANY(%(sections)s)" in sql_call
    assert "ORDER BY score DESC" in sql_call
