import json
import logging
from typing import List, Optional, Union, Any, Dict
import psycopg
from psycopg.rows import dict_row
from pgvector import Vector
from pgvector.psycopg import register_vector

from storage.base import VectorStore, SearchResult, FilterCriteria
from storage.db import get_connection
from document_processing.models.document import ExtractedDocument, StructuredDocument
from document_processing.models.chunk import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

class PostgresVectorStore(VectorStore):
    """
    Production storage implementation backed by PostgreSQL and pgvector.
    Provides vector similarity search, relational academic metadata, and full-text search.
    """

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string

    def _get_conn(self) -> psycopg.Connection:
        conn = get_connection(self.connection_string)
        return conn

    def save_document(self, document: Union[ExtractedDocument, StructuredDocument]) -> None:
        """
        Inserts or updates document record with metadata.
        """
        meta = document.metadata
        extra_json = json.dumps(meta.extra)

        sql = """
        INSERT INTO documents (
            document_id, source_path, filename, title, authors, 
            publication_year, journal, doi, source_url, extra_metadata
            , source_type, license, access_status, content_hash, file_size_bytes,
            ingestion_time, extraction_status
        ) VALUES (
            %(doc_id)s, %(source_path)s, %(filename)s, %(title)s, %(authors)s,
            %(year)s, %(journal)s, %(doi)s, %(source_url)s, %(extra)s::jsonb,
            %(source_type)s, %(license)s, %(access_status)s, %(content_hash)s,
            %(file_size_bytes)s, %(ingestion_time)s, %(extraction_status)s
        )
        ON CONFLICT (document_id) DO UPDATE SET
            source_path = EXCLUDED.source_path,
            filename = EXCLUDED.filename,
            title = EXCLUDED.title,
            authors = EXCLUDED.authors,
            publication_year = EXCLUDED.publication_year,
            journal = EXCLUDED.journal,
            doi = EXCLUDED.doi,
            source_url = EXCLUDED.source_url,
            source_type = EXCLUDED.source_type,
            license = EXCLUDED.license,
            access_status = EXCLUDED.access_status,
            content_hash = EXCLUDED.content_hash,
            file_size_bytes = EXCLUDED.file_size_bytes,
            ingestion_time = EXCLUDED.ingestion_time,
            extraction_status = EXCLUDED.extraction_status,
            extra_metadata = EXCLUDED.extra_metadata;
        """

        params = {
            "doc_id": document.document_id,
            "source_path": document.source_path,
            "filename": document.filename,
            "title": meta.title,
            "authors": meta.authors,
            "year": meta.publication_year,
            "journal": meta.journal,
            "doi": meta.doi,
            "source_url": meta.source_url,
            "source_type": meta.source_type,
            "license": meta.license,
            "access_status": meta.access_status,
            "content_hash": meta.content_hash,
            "file_size_bytes": meta.file_size_bytes,
            "ingestion_time": meta.ingestion_time,
            "extraction_status": meta.extraction_status,
            "extra": extra_json
        }

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        logger.debug(f"Document {document.document_id} saved successfully.")

    def save_chunks(self, chunks: List[Chunk]) -> None:
        """
        Persists chunks and high-dimensional embeddings in batches.
        """
        if not chunks:
            return

        sql = """
        INSERT INTO chunks (
            chunk_id, document_id, chunk_index, section, page_start, page_end,
            content, extra_metadata, embedding
            , parent_id, chunking_strategy, chunking_version, character_count,
            token_count, embedding_model, embedding_dimensions, embedding_normalized
        ) VALUES (
            %(chunk_id)s, %(document_id)s, %(chunk_index)s, %(section)s, 
            %(page_start)s, %(page_end)s, %(content)s, %(extra_metadata)s::jsonb, %(embedding)s,
            %(parent_id)s, %(chunking_strategy)s, %(chunking_version)s, %(character_count)s,
            %(token_count)s, %(embedding_model)s, %(embedding_dimensions)s, %(embedding_normalized)s
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            document_id = EXCLUDED.document_id,
            chunk_index = EXCLUDED.chunk_index,
            section = EXCLUDED.section,
            page_start = EXCLUDED.page_start,
            page_end = EXCLUDED.page_end,
            content = EXCLUDED.content,
            parent_id = EXCLUDED.parent_id,
            chunking_strategy = EXCLUDED.chunking_strategy,
            chunking_version = EXCLUDED.chunking_version,
            character_count = EXCLUDED.character_count,
            token_count = EXCLUDED.token_count,
            embedding_model = EXCLUDED.embedding_model,
            embedding_dimensions = EXCLUDED.embedding_dimensions,
            embedding_normalized = EXCLUDED.embedding_normalized,
            extra_metadata = EXCLUDED.extra_metadata,
            embedding = EXCLUDED.embedding;
        """

        rows = []
        for c in chunks:
            if c.embedding is not None and len(c.embedding) != 768:
                raise ValueError(
                    f"Chunk {c.chunk_id} has {len(c.embedding)} embedding dimensions; "
                    "the current PostgreSQL index requires 768."
                )
            emb = Vector(c.embedding) if c.embedding is not None else None
            meta_dict = c.metadata.extra.copy() if c.metadata else {}
            if c.metadata and c.metadata.document_title:
                meta_dict["document_title"] = c.metadata.document_title

            rows.append({
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "chunk_index": c.chunk_index,
                "section": c.metadata.section if c.metadata else None,
                "page_start": c.metadata.page_start if c.metadata else None,
                "page_end": c.metadata.page_end if c.metadata else None,
                "content": c.content,
                "parent_id": c.parent_id,
                "chunking_strategy": c.chunking_strategy,
                "chunking_version": c.chunking_version,
                "character_count": c.character_count or len(c.content),
                "token_count": c.token_count or len(c.content.split()),
                "embedding_model": getattr(self, "embedding_model", None),
                "embedding_dimensions": len(c.embedding) if c.embedding is not None else None,
                "embedding_normalized": getattr(self, "embedding_normalized", None),
                "extra_metadata": json.dumps(meta_dict),
                "embedding": emb
            })

        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
        logger.info(f"Saved {len(chunks)} chunks to PostgreSQL.")

    def search_vector(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """
        Finds nearest chunks using pgvector cosine distance (<=>).
        Returns results ordered by similarity score descending (1.0 = identical).
        """
        where_clauses = ["c.embedding IS NOT NULL"]
        params: Dict[str, Any] = {
            "query_emb": Vector(query_embedding),
            "limit": top_k
        }

        if filter_criteria:
            if filter_criteria.document_ids:
                where_clauses.append("c.document_id = ANY(%(doc_ids)s)")
                params["doc_ids"] = filter_criteria.document_ids
            if filter_criteria.sections:
                where_clauses.append("c.section = ANY(%(sections)s)")
                params["sections"] = filter_criteria.sections
            if filter_criteria.min_year is not None:
                where_clauses.append("d.publication_year >= %(min_year)s")
                params["min_year"] = filter_criteria.min_year
            if filter_criteria.max_year is not None:
                where_clauses.append("d.publication_year <= %(max_year)s")
                params["max_year"] = filter_criteria.max_year
            if filter_criteria.authors_contain:
                where_clauses.append("array_to_string(d.authors, ' ') ILIKE %(author)s")
                params["author"] = f"%{filter_criteria.authors_contain}%"
            if filter_criteria.embedding_model:
                where_clauses.append("c.embedding_model = %(embedding_model)s")
                params["embedding_model"] = filter_criteria.embedding_model

        where_sql = " AND ".join(where_clauses)

        # 1 - (c.embedding <=> %(query_emb)s) gives cosine similarity in [-1, 1] (typically [0, 1])
        sql = f"""
        SELECT 
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.section,
            c.page_start,
            c.page_end,
            c.extra_metadata,
            d.title AS document_title,
            d.authors,
            d.publication_year,
            d.journal,
            d.doi,
            d.source_url,
            1.0 - (c.embedding <=> %(query_emb)s) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.document_id
        WHERE {where_sql}
        ORDER BY c.embedding <=> %(query_emb)s ASC
        LIMIT %(limit)s;
        """

        results: List[SearchResult] = []
        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    extra = row["extra_metadata"] or {}
                    results.append(SearchResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        chunk_index=row["chunk_index"],
                        content=row["content"],
                        section=row["section"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        score=float(row["score"]),
                        document_title=row["document_title"],
                        authors=row.get("authors"),
                        publication_year=row.get("publication_year"),
                        journal=row.get("journal"),
                        doi=row.get("doi"),
                        source_url=row.get("source_url"),
                        metadata=extra
                    ))
        return results

    def search_text(
        self,
        query_text: str,
        top_k: int = 5,
        filter_criteria: Optional[FilterCriteria] = None
    ) -> List[SearchResult]:
        """
        Executes full-text lexical search using PostgreSQL's tsvector / plainto_tsquery.
        """
        where_clauses = ["c.tsv @@ plainto_tsquery('english', %(query_text)s)"]
        params: Dict[str, Any] = {
            "query_text": query_text,
            "limit": top_k
        }

        if filter_criteria:
            if filter_criteria.document_ids:
                where_clauses.append("c.document_id = ANY(%(doc_ids)s)")
                params["doc_ids"] = filter_criteria.document_ids
            if filter_criteria.sections:
                where_clauses.append("c.section = ANY(%(sections)s)")
                params["sections"] = filter_criteria.sections
            if filter_criteria.min_year is not None:
                where_clauses.append("d.publication_year >= %(min_year)s")
                params["min_year"] = filter_criteria.min_year
            if filter_criteria.max_year is not None:
                where_clauses.append("d.publication_year <= %(max_year)s")
                params["max_year"] = filter_criteria.max_year
            if filter_criteria.authors_contain:
                where_clauses.append("array_to_string(d.authors, ' ') ILIKE %(author)s")
                params["author"] = f"%{filter_criteria.authors_contain}%"

        where_sql = " AND ".join(where_clauses)

        sql = f"""
        SELECT 
            c.chunk_id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.section,
            c.page_start,
            c.page_end,
            c.extra_metadata,
            d.title AS document_title,
            d.authors,
            d.publication_year,
            d.journal,
            d.doi,
            d.source_url,
            ts_rank_cd(c.tsv, plainto_tsquery('english', %(query_text)s)) AS score
        FROM chunks c
        JOIN documents d ON c.document_id = d.document_id
        WHERE {where_sql}
        ORDER BY score DESC
        LIMIT %(limit)s;
        """

        results: List[SearchResult] = []
        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                for row in rows:
                    extra = row["extra_metadata"] or {}
                    results.append(SearchResult(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        chunk_index=row["chunk_index"],
                        content=row["content"],
                        section=row["section"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        score=float(row["score"]),
                        document_title=row["document_title"],
                        authors=row.get("authors"),
                        publication_year=row.get("publication_year"),
                        journal=row.get("journal"),
                        doi=row.get("doi"),
                        source_url=row.get("source_url"),
                        metadata=extra
                    ))
        return results

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        Retrieves a single chunk by ID.
        """
        sql = """
        SELECT chunk_id, document_id, chunk_index, section, page_start, page_end,
               content, extra_metadata, embedding
        FROM chunks
        WHERE chunk_id = %(chunk_id)s;
        """
        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, {"chunk_id": chunk_id})
                row = cur.fetchone()
                if not row:
                    return None
                
                extra = row["extra_metadata"] or {}
                emb = row["embedding"]
                embedding_list = emb.to_list() if hasattr(emb, "to_list") else (list(emb) if emb is not None else None)

                meta = ChunkMetadata(
                    document_title=extra.get("document_title"),
                    section=row["section"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    extra=extra
                )
                return Chunk(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    chunk_index=row["chunk_index"],
                    content=row["content"],
                    metadata=meta,
                    embedding=embedding_list
                )

    def get_chunks_for_document(self, document_id: str) -> List[Chunk]:
        """
        Retrieves all chunks for a document ordered by chunk_index.
        """
        sql = """
        SELECT chunk_id, document_id, chunk_index, section, page_start, page_end,
               content, extra_metadata, embedding
        FROM chunks
        WHERE document_id = %(doc_id)s
        ORDER BY chunk_index ASC;
        """
        chunks: List[Chunk] = []
        with self._get_conn() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(sql, {"doc_id": document_id})
                for row in cur.fetchall():
                    extra = row["extra_metadata"] or {}
                    emb = row["embedding"]
                    embedding_list = emb.to_list() if hasattr(emb, "to_list") else (list(emb) if emb is not None else None)
                    meta = ChunkMetadata(
                        document_title=extra.get("document_title"),
                        section=row["section"],
                        page_start=row["page_start"],
                        page_end=row["page_end"],
                        extra=extra
                    )
                    chunks.append(Chunk(
                        chunk_id=row["chunk_id"],
                        document_id=row["document_id"],
                        chunk_index=row["chunk_index"],
                        content=row["content"],
                        metadata=meta,
                        embedding=embedding_list
                    ))
        return chunks

    def delete_document(self, document_id: str) -> bool:
        """
        Deletes document and cascades to chunks.
        """
        sql = "DELETE FROM documents WHERE document_id = %(doc_id)s;"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"doc_id": document_id})
                deleted = cur.rowcount > 0
        return deleted

    def count_documents(self) -> int:
        sql = "SELECT COUNT(*) FROM documents;"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()[0]

    def count_chunks(self) -> int:
        sql = "SELECT COUNT(*) FROM chunks;"
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                return cur.fetchone()[0]
