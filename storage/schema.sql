-- PhysioEvidence PostgreSQL Schema with pgvector
-- Enables vector similarity search, relational academic metadata, and full-text search

CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Documents table: Stores metadata about research papers
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(64) PRIMARY KEY,
    source_path TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT,
    authors TEXT[],
    publication_year INTEGER,
    journal TEXT,
    doi TEXT,
    source_url TEXT,
    source_type TEXT,
    license TEXT,
    access_status TEXT,
    content_hash VARCHAR(64),
    file_size_bytes BIGINT,
    ingestion_time TIMESTAMPTZ,
    extraction_status TEXT NOT NULL DEFAULT 'extracted',
    extra_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Chunks table: Stores text chunks, section context, and high-dimensional vector embeddings
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(64) PRIMARY KEY,
    document_id VARCHAR(64) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    section TEXT,
    page_start INTEGER,
    page_end INTEGER,
    content TEXT NOT NULL,
    parent_id VARCHAR(64),
    chunking_strategy TEXT NOT NULL DEFAULT 'section-recursive-char-v1',
    chunking_version TEXT NOT NULL DEFAULT '1',
    character_count INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    embedding_model TEXT,
    embedding_dimensions INTEGER,
    embedding_normalized BOOLEAN,
    extra_metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(768),
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Performance & Query Indexes
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(publication_year);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin(tsv);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash) WHERE content_hash IS NOT NULL;

-- HNSW Vector Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
ON chunks USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);
