# ADR 0001: PostgreSQL + pgvector for Unified Evidence and Vector Storage

## Context
PhysioEvidence requires persistent storage for ingested documents, hierarchical text chunks, academic metadata (authors, journal, publication year, DOI, section titles), and high-dimensional vector embeddings (768 dimensions for Gemini `text-embedding-004`). In future phases, the system will support hybrid retrieval (combining keyword/lexical search and vector similarity) as well as metadata-filtered queries (e.g., searching specifically within "Methods" or papers published after 2020).

## Decision
We choose **PostgreSQL with the `pgvector` extension** using `psycopg` (v3) binary for the storage layer.

The schema establishes:
1. `documents` table: Primary key `document_id` (UUID/string), source path, filename, title, authors (text array), publication year, journal, DOI, source URL, and JSONB `extra_metadata`.
2. `chunks` table: Primary key `chunk_id` (UUID/string), foreign key to `documents.document_id` (ON DELETE CASCADE), chunk index, section name, page start/end, text content, JSONB `extra_metadata`, and a 768-dimensional `vector` column for embeddings.
3. Indexes:
   - HNSW index on `chunks.embedding` using cosine distance (`vector_cosine_ops`) for sub-linear similarity search.
   - B-tree indexes on `chunks.document_id`, `chunks.section`, `documents.publication_year` for fast filtered queries.
   - PostgreSQL `tsvector` generated column / GIN index for native BM25/full-text search (ready for Phase 3 hybrid search without external search engines).

## Alternatives Considered
1. **Dedicated Vector Databases (Pinecone, Qdrant, Milvus)**:
   - *Pros*: Out-of-the-box hosted vector scaling.
   - *Cons*: Introduces dual-storage complexity (documents/relational metadata in one place, vectors in another), eventual consistency issues, network overhead, and vendor lock-in.
2. **Embedded Vector Stores (ChromaDB, LanceDB)**:
   - *Pros*: Quick local file-based prototyping.
   - *Cons*: Limited concurrency, weaker relational capabilities for structured joins and metadata filtering, and less suited for server deployments.
3. **SQLite + sqlite-vss**:
   - *Pros*: Lightweight local database.
   - *Cons*: Lacks production-grade full-text + vector fusion, weaker concurrent write handling, and complex native extensions on Windows.

## Consequences
- **Positive**: Single ACID-compliant database for relational paper metadata, structured text chunks, vector embeddings, and full-text search. Zero synchronization lag between metadata and vectors.
- **Positive**: Direct support for hybrid retrieval via SQL (combining full-text search with vector cosine distance using Reciprocal Rank Fusion or linear combination).
- **Negative / Considerations**: Requires PostgreSQL instance with `pgvector` extension installed. Development and test environments should provide an abstraction layer or containerized PostgreSQL instance, and unit tests must be runnable using mock/in-memory abstractions when PostgreSQL is offline.
