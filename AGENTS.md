# PhysioEvidence

## What this is
PhysioEvidence is an evidence-grounded research assistant designed to help physiotherapists explore scientific literature, retrieve evidence, and generate answers strictly grounded in clinical research papers with verifiable source citations.

## Stack
- Python 3.14
- Models: Pydantic v2
- PDF Extraction: PyMuPDF (fitz)
- Embeddings: Local Ollama (`embeddinggemma-2:270m`, 768 dimensions) or Google Gemini (`text-embedding-004`, 768 dimensions)
- Storage & Vector DB: PostgreSQL 16+ with `pgvector`
- Database driver: `psycopg` (v3) binary
- Generation: Google Gemini (`gemini-3.5-flash-lite`)

- Testing: `pytest`
- Observability: LangSmith (optional tracing via environment variables)

## Commands
- Install dependencies: `.venv\Scripts\pip install -r requirements.txt`
- Run test suite: `.venv\Scripts\pytest`
- Run document processing CLI (single file):
  `.venv\Scripts\python -m document_processing.cli --input <pdf_path> [--output <json_path>] [--no-embeddings] [--save-db]`
- Ingest a folder of PDFs:
  `.venv\Scripts\python -m document_processing.cli --input-dir <dir_path> [--recursive] [--save-db]`
- Initialize PostgreSQL schema:
  `.venv\Scripts\python -m document_processing.cli --init-db`
- Ask a clinical question (RAG):
  `.venv\Scripts\python -m rag.cli --question "What quadriceps symmetry index is needed after ACL reconstruction?" [--hybrid] [--rerank]`
- Run retrieval benchmark:
  `.venv\Scripts\python -m evaluation.run_benchmark`

## Conventions
- `document_processing/`: Ingestion, extraction, cleaning, section detection, chunking, and embedding.
- `storage/`: Database connection, schema migrations (`schema.sql`), and vector store implementations (`PostgresVectorStore`, `InMemoryVectorStore`).
- `retrieval/`: Semantic vector retrieval (`VectorRetriever`), hybrid search with Reciprocal Rank Fusion (`HybridRetriever`), and reranking (`CrossEncoderReranker`).
- `generation/`: Grounded prompt engineering, context assembly, source citation mapping, and LLM synthesis (`EvidenceSynthesizer`).
- `rag/`: End-to-end RAG orchestrator (`PhysioEvidenceRAG`) and question CLI (`rag.cli`).
- `evaluation/`: Curated benchmark questions, evaluation corpus, metrics (`recall_at_k`, `mrr`, `ndcg_at_k`), and benchmark runner.
- `observability/`: Non-invasive LangSmith tracing wrapper (`trace_operation`).
- Keep external services (Gemini, LangSmith, PostgreSQL) behind abstractions so unit tests run fast without live credentials.

## Treat as high-stakes (explain before implementing)
- Medical groundedness: Never allow the LLM to hallucinate or fabricate clinical advice; enforce refusal on insufficient evidence.
- Database schema changes and pgvector index configurations (vector dimensions, distance metrics, HNSW parameters).
- Embedding dimensionality consistency across chunk generation and queries (768 dimensions).
- Safe handling of credentials and API keys: NEVER commit `.env` or secret keys.