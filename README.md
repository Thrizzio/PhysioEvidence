# PhysioEvidence
PhysioEvidence is a RAG-based research assistant designed to help physiotherapists
explore scientific literature and retrieve evidence relevant to clinical questions.
**The core idea:** Find the right research, retrieve the right parts of it, and generate
answers grounded in the original sources.
## What the system does
Research Papers
 ↓
Document Ingestion
 ↓
Text Extraction & Cleaning
 ↓
Section Detection
 ↓
Context-Aware Chunking
 ↓
Embeddings
 ↓
PostgreSQL + pgvector
 ↓
Retrieval
 ↓
Reranking
 ↓
LLM
 ↓
Source-Grounded Answer
The current focus is the document-processing and ingestion pipeline. Retrieval and
generation components will be added incrementally.
## Current Implementation
### Document Processing
Research papers are processed using PyMuPDF for PDF text extraction, Pydantic for
structured document models, deterministic text cleaning and normalization, section
detection for common research-paper structures, and section-aware chunking while
preserving document context.
The goal is to avoid treating an entire research paper as an undifferentiated block of
text.
Paper
■■■ Abstract
■■■ Introduction
■■■ Methods
■■■ Results
■■■ Discussion
■■■ References
## Planned Retrieval Pipeline
### 1. Semantic Retrieval
Documents will be embedded and stored using Gemini embeddings, PostgreSQL, and pgvector.
A user's question can then retrieve chunks based on semantic similarity rather than exact
keyword matches.
### 2. Hybrid Retrieval
Combine semantic/vector search with traditional keyword-based retrieval. This is useful
when exact terminology matters, particularly for medical and scientific literature.
### 3. Parent-Child Chunking
The system will experiment with smaller retrieval chunks while retaining larger
contextual units. A small child chunk can provide precise retrieval while corresponding
parent context can be supplied to the LLM to reduce loss of surrounding information.
### 4. Multi-Query Retrieval
A single user question can be rewritten into multiple search queries. This can help when
a question can be expressed using different terminology.
### 5. Reranking
Initial retrieval can return a larger candidate set. A reranking stage can then evaluate
candidates more carefully and pass the most relevant results to the LLM.
### 6. Metadata Filtering
Research papers contain useful metadata such as authors, publication year, journal, DOI,
source, and research section. Metadata can be used to narrow retrieval before semantic
search or alongside it.
## Answer Generation
The final language model should answer questions using retrieved research rather than
relying solely on its pretrained knowledge. The intended flow is:
Question
 ↓
Retrieve Evidence
 ↓
Select Relevant Context
 ↓
LLM
 ↓
Answer + Source References
The system should be able to identify which research was used to support an answer.
## Technology
**Current / Planned Stack**
• Python
• FastAPI
• LangChain
• Gemini
• PostgreSQL
• pgvector
• PyMuPDF
• Pydantic
• Docker
## Development Roadmap
### Phase 1 — Document Processing
✓ Project structure • ✓ Pydantic document models • ✓ PDF extraction • ✓ Basic text
cleaning • ✓ Section detection • ✓ Context-aware chunking • ✓ Processing tests
### Phase 2 — Embeddings & Storage
✓ Gemini embedding integration • ✓ PostgreSQL schema • ✓ pgvector integration • ✓
Document and chunk indexing • ✓ In-memory vector store
### Phase 3 — Retrieval
✓ Semantic retrieval • ✓ Metadata filtering • ✓ Hybrid retrieval (Reciprocal Rank Fusion) •
✓ Cross-encoder/heuristic reranking
### Phase 4 — Generation & Evaluation
✓ Source-grounded generation • ✓ Citation handling ([Source N]) • ✓ Refusal on insufficient evidence •
✓ Retrieval evaluation suite • ✓ Benchmark comparison CLI

## Why this project?
Scientific literature is continuously growing, while finding the relevant evidence for a
specific question can require searching across many papers. PhysioEvidence explores how
modern retrieval techniques can be combined to make large collections of physiotherapy
research easier to search and understand while keeping generated answers connected to
their underlying sources.

## Core Modules & Architecture

### 1. Document Processing (`document_processing/`)
- Extracts PDF text page-by-page using PyMuPDF.
- Cleans whitespace and repairs line-broken hyphenated words deterministically.
- Detects standard scientific paper sections (Abstract, Introduction, Methods, Results, Discussion, etc.).
- Hierarchically chunks text by section, paragraph, and sentence boundaries.
- Generates 768-dimensional vector embeddings with Gemini `text-embedding-004`.

### 2. Storage & Vector DB (`storage/`)
- Relational PostgreSQL schema with `pgvector` extension.
- HNSW index on chunk embeddings for fast cosine similarity search.
- GIN index on text search vectors (`tsvector`) for full-text search.
- In-memory vector store fallback for instant offline execution and testing.

### 3. Retrieval (`retrieval/`)
- **Semantic Vector Search**: Asymmetric query embedding with cosine distance ranking.
- **Hybrid Retrieval**: Combines dense vector search and sparse full-text search via **Reciprocal Rank Fusion (RRF)**.
- **Reranker**: Second-stage scoring with empirical section weighting (Results, Conclusions) and exact term proximity.

### 4. Generation & Attribution (`generation/`)
- Strictly grounded clinical synthesis using Google Gemini.
- Mandatory source citations (`[Source 1]`) mapped directly to document title, section, and page numbers.
- Explicit refusal when evidence is absent or insufficient to answer the clinical question.

### 5. Evaluation & Benchmarking (`evaluation/`)
- Benchmark dataset of realistic physiotherapy clinical questions with curated ground truth.
- Computes standard information retrieval metrics: **Recall@K**, **Precision@K**, **MRR**, and **nDCG@K**.

### 6. Observability (`observability/`)
- Non-invasive LangSmith tracing integration for inspecting questions, retrieved chunks, reranking scores, latencies, and token consumption. Enabled via `LANGSMITH_TRACING=true`.

---

## Installation & Setup

1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env` (refer to `.env.example`):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=physioevidence
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
```

3. (Optional) Initialize the PostgreSQL database schema:
```bash
python -m document_processing.cli --init-db
```

---

## Usage Guide

### Ingesting Documents

Process a single PDF:
```bash
python -m document_processing.cli --input data/sample_paper.pdf --save-db
```

Batch-ingest an entire directory of PDFs:
```bash
python -m document_processing.cli --input-dir data/papers/ --recursive --save-db
```

Inspect chunking without generating embeddings (dry run):
```bash
python -m document_processing.cli --input data/sample_paper.pdf --output chunks.json --no-embeddings
```

### Asking Clinical Questions (RAG)

Run an evidence-grounded query with hybrid retrieval and reranking:
```bash
python -m rag.cli --question "What quadriceps symmetry index is recommended before return to sport after ACL reconstruction?" --hybrid --rerank
```

Filter by specific paper sections or publication year:
```bash
python -m rag.cli --question "What are the outcomes of heavy slow resistance training in Achilles tendinopathy?" --section Results --min-year 2020
```

### Running Retrieval Benchmarks

Evaluate and compare retrieval strategies (Pure Vector vs Hybrid RRF vs Reranking) across the curated clinical evaluation dataset:
```bash
python -m evaluation.run_benchmark
```

### Running Test Suite

Run all 45 automated unit and component tests:
```bash
pytest
```
