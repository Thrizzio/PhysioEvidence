# PhysioEvidence

## What this is
PhysioEvidence is a RAG-based research assistant designed to help physiotherapists explore scientific literature and retrieve evidence grounded in clinical papers.

## Stack
- Python 3.14
- Models: Pydantic v2
- PDF Extraction: PyMuPDF (fitz)
- Embeddings: Google Gemini (`text-embedding-004`, 768 dimensions)
- Storage & Vector DB: PostgreSQL 16+ with `pgvector`
- Database driver: `psycopg` (v3) binary
- Testing: `pytest`
- Observability: LangSmith (optional tracing via environment variables)

## Commands
- Install dependencies: `.venv\Scripts\pip install -r requirements.txt`
- Run tests: `.venv\Scripts\pytest`
- Run document processing CLI: `.venv\Scripts\python -m document_processing.cli --input <pdf_path> --output <output_json>`

## Conventions
- `document_processing/`: Ingestion, extraction, cleaning, section detection, chunking, and embedding.
  - `models/`: Pydantic domain models for documents, pages, sections, and chunks.
  - `loaders/`: Document loaders with abstract base classes (`DocumentLoader`).
  - `processors/`: Deterministic text cleaners, section detectors, and hierarchical chunkers.
  - `embeddings/`: Embedding provider abstraction and concrete implementations.
  - `pipeline/`: Orchestrating pipelines from raw file to embedded chunks.
  - `tests/`: Component and unit tests.
- `storage/`: Database connection, schema migrations, and vector storage repositories.
- Keep external services (Gemini, LangSmith, PostgreSQL) behind abstractions so unit tests run fast without live credentials.

## Treat as high-stakes (explain before implementing)
- Database schema changes and pgvector index configurations (vector dimensions, index types like HNSW/IVFFlat, indexing distance metrics).
- Embedding dimensionality consistency across chunk generation and queries.
- Safe handling of credentials and API keys: NEVER commit `.env` or secret keys.