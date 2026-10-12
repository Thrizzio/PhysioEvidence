# Architecture

## Status
Production Foundation Implemented — backend phases 1-5 active, with separate
public research and private ingestion service boundaries:
- Phase 1: Document Processing (PDF extraction, text cleaning, section detection, context-aware chunking).
- Phase 2: Embeddings & Storage (PostgreSQL + pgvector, schema, HNSW index, in-memory test store).
- Phase 3: Retrieval (Dense vector retrieval, Hybrid search with Reciprocal Rank Fusion, section-weighted Reranker).
- Phase 4: Generation (Context assembly with budget limits, strict source attribution `[Source N]`, refusal on insufficient evidence, Gemini LLM).
- Phase 5: Evaluation (Benchmark dataset of clinical queries, Recall@K, MRR, nDCG@K).
- Observability: Non-invasive LangSmith tracing integration.
- Websites: `web/research/` contains only the public Q&A UI; `web/admin/` is a
  separate corpus-owner upload/status UI protected by the ingestion API.

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
  - `base.py`: `EmbeddingProvider` abstract interface (`embed_chunks`, `embed_query`).
  - `gemini_embeddings.py`: `GeminiEmbeddingProvider` batching chunk embedding generation using Gemini `text-embedding-004` (768 dimensions) with asymmetric retrieval task types (`retrieval_document` and `retrieval_query`).
- **`pipeline/`**:
  - `document_pipeline.py`: Orchestrates loading, section structuring, chunking, embedding, and vector store persistence. Includes `process_directory()` for batch folder ingestion.
- **`cli.py`**: Command-line interface supporting single PDF and batch folder ingestion (`--input`, `--input-dir`, `--save-db`, `--init-db`).

### 2. Storage & Vector DB (`storage/`)
- **`base.py`**: Abstract `VectorStore` interface, `SearchResult`, and `FilterCriteria` models.
- **`schema.sql`**: Relational schema with `pgvector` extension for `documents` and `chunks`, including HNSW vector index (`vector_cosine_ops`) and GIN index on `tsvector` for full-text search.
- **`postgres_store.py`**: Concrete `PostgresVectorStore` using `psycopg` and `pgvector`.
- **`in_memory_store.py`**: High-performance in-memory vector store for unit tests, offline development, and deterministic benchmark comparisons.

### 3. Retrieval (`retrieval/`)
- **`base.py`**: Abstract `BaseRetriever` interface.
- **`vector_retriever.py`**: Dense semantic retriever embedding queries and performing cosine similarity search.
- **`hybrid_retriever.py`**: Combines dense vector search and sparse full-text search using **Reciprocal Rank Fusion (RRF)** ($k=60$).
- **`reranker.py`**: `CrossEncoderReranker` applying section-weight boosts (Results, Conclusions) and exact phrase coverage.

### 4. Generation & Synthesis (`generation/`)
- **`context_assembler.py`**: `ContextAssembler` formatting evidence chunks into structured, numbered passages (`[Source 1]`) within strict token/character budgets.
- **`prompt_templates.py`**: Clinical system prompts enforcing strict groundedness, mandatory source citations, and explicit refusals on insufficient evidence.
- **`llm.py`**: `LLMProvider` abstraction with `GeminiLLMProvider` and `MockLLMProvider`.
- **`synthesizer.py`**: `EvidenceSynthesizer` generating verified, citation-mapped `EvidenceAnswer` objects.

### 5. Unified RAG Pipeline (`rag/`)
- **`pipeline.py`**: `PhysioEvidenceRAG` composing retriever, optional reranker, and evidence synthesizer.
- **`cli.py`**: Terminal CLI for asking clinical questions with `--hybrid`, `--rerank`, and metadata filters.

### 6. Evaluation (`evaluation/`)
- **`dataset.py`**: Curated benchmark dataset of 6 realistic physiotherapy clinical questions with expected chunk IDs and ground-truth clinical answers.
- **`metrics.py`**: Industry-standard retrieval metrics: `recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`.
- **`retrieval_evaluator.py`**: `RetrievalEvaluator` for comparative benchmarking across strategies with automated markdown reports.
- **`run_benchmark.py`**: CLI script (`python -m evaluation.run_benchmark`) executing live strategy comparisons.

### 7. Observability (`observability/`)
- Non-invasive tracing hooks in `tracing.py` that integrate with LangSmith when `LANGSMITH_TRACING=true` and credentials are provided, running with zero overhead when disabled.

### 8. Service boundaries (`services/`)
- **`research_api/`**: Public query endpoint (`POST /api/query`) and health
  endpoint. It never accepts document uploads.
- **`ingestion_service/`**: Private upload, job status, document list, and
  storage usage endpoints. It validates actual PDF parser behavior, applies
  content-hash idempotency, and calls the canonical document pipeline.
- Both services build on shared domain abstractions rather than duplicating
  extraction, chunking, storage, or generation logic.

## Data Flow

```
1. Document Ingestion Flow:
   PDF File(s)
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
   [PostgresVectorStore (pgvector)] ──> Persistent relational, full-text & vector storage

2. Question-Answering (RAG) Flow:
   User Clinical Question
      │
      ▼
   [Retriever (Vector / Hybrid RRF)] ──> Candidate SearchResults
      │
      ▼
   [(Optional) CrossEncoderReranker] ──> Top-K Reranked Evidence
      │
      ▼
   [ContextAssembler] ──> Formatted Numbered Passages ([Source 1], [Source 2])
      │
      ▼
   [EvidenceSynthesizer + Gemini LLM] ──> EvidenceAnswer (Grounded Response + Sources)
```

## Key Decisions
See `docs/adr/` for the reasoning behind architectural decisions:
- `docs/adr/0001-postgresql-pgvector-storage-architecture.md`: PostgreSQL + pgvector for unified relational, full-text, and vector storage.
- `docs/adr/0002-hybrid-retrieval-with-reciprocal-rank-fusion.md`: Hybrid Retrieval with Reciprocal Rank Fusion (RRF).
- `docs/adr/0003-grounded-generation-and-citation-architecture.md`: Grounded generation, citation mapping, and refusal on insufficient evidence.
