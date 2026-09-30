# Architecture

## Status
Active Development — Phase 1 (Document Processing) is complete and tested. Phase 2 (Embeddings & PostgreSQL + pgvector Storage) is underway.

## Modules

### 1. Document Processing (`document_processing/`)
- **`models/`**: Strongly-typed Pydantic models.
  - `document.py`: `ExtractedDocument`, `Page`, `DocumentMetadata`, `Section`, `StructuredDocument`.
  - `chunk.py`: `Chunk`, `ChunkMetadata` (retains document ID, title, section, page numbers, and optional embedding vector).
- **`loaders/`**: Document extraction interface and implementations.
  - `base.py`: `DocumentLoader` abstract interface.
  - `pdf_loader.py`: `PDFLoader` using PyMuPDF (`fitz`), preserving page breaks and document metadata.
- **`processors/`**: Deterministic content refinement.
  - `cleaner.py`: `TextCleaner` removes form feeds, fixes hyphenation across line breaks, and collapses abnormal whitespace.
  - `section_detector.py`: `SectionDetector` detects canonical academic sections (Abstract, Introduction, Methods, Results, Discussion, References, etc.) using regex and alias mapping.
  - `chunker.py`: `SectionAwareChunker` chunks hierarchically by section, paragraph, and sentence rather than fixed arbitrary character cuts.
- **`embeddings/`**:
  - `base.py`: `EmbeddingProvider` abstract interface.
  - `gemini_embeddings.py`: `GeminiEmbeddingProvider` batching chunk embedding generation using Gemini `text-embedding-004` (768 dimensions).
- **`pipeline/`**:
  - `document_pipeline.py`: Orchestrates loading, section structuring, chunking, and embedding with timing and error wrapping (`DocumentProcessingError`).
- **`cli.py`**: Command-line interface for local document processing.

### 2. Storage & Vector DB (`storage/`)
- **`schema.sql`**: Relational schema with `pgvector` extension for `documents` and `chunks`.
- **`postgres_store.py`**: Repository implementing vector store operations, cosine distance similarity search, metadata filtering, and upserts.

### 3. Observability (`observability/`)
- Non-invasive tracing hooks for LangSmith that activate only when `LANGSMITH_TRACING=true` and required credentials are set. Basic pipeline runs with zero overhead when disabled.

## Data Flow

```
PDF File
   │
   ▼
[PDFLoader (PyMuPDF)] ──> ExtractedDocument (pages preserved)
   │
   ▼
[SectionDetector + TextCleaner] ──> StructuredDocument (canonical sections)
   │
   ▼
[SectionAwareChunker] ──> List[Chunk] (hierarchical chunk boundaries)
   │
   ▼
[GeminiEmbeddingProvider] ──> List[Chunk] with 768d vectors
   │
   ▼
[PostgresVectorStore (pgvector)] ──> Persistent relational & vector storage
```

## Key Decisions
See `docs/adr/` for the reasoning behind architectural decisions:
- `docs/adr/0001-postgresql-pgvector-storage-architecture.md`: PostgreSQL + pgvector for unified relational, full-text, and vector storage.