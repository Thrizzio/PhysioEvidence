-- Forward-compatible migration for databases created before processing metadata
-- became explicit. Safe to run repeatedly.
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source_type TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS license TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS access_status TEXT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_size_bytes BIGINT;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS ingestion_time TIMESTAMPTZ;
ALTER TABLE documents ADD COLUMN IF NOT EXISTS extraction_status TEXT NOT NULL DEFAULT 'extracted';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_id VARCHAR(64);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunking_strategy TEXT NOT NULL DEFAULT 'section-recursive-char-v1';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS chunking_version TEXT NOT NULL DEFAULT '1';
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS character_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS token_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_model TEXT;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_dimensions INTEGER;
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_normalized BOOLEAN;
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash) WHERE content_hash IS NOT NULL;
