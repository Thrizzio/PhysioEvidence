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
cleaning • ■ Section detection • ■ Context-aware chunking • ■ Processing tests
### Phase 2 — Embeddings & Storage
■ Gemini embedding integration • ■ PostgreSQL schema • ■ pgvector integration • ■
Document and chunk indexing
### Phase 3 — Retrieval
■ Semantic retrieval • ■ Metadata filtering • ■ Hybrid retrieval • ■ Parent-child
retrieval • ■ Multi-query retrieval • ■ Reranking
### Phase 4 — Generation & Evaluation
■ Source-grounded generation • ■ Citation handling • ■ Retrieval evaluation • ■ Answer
quality evaluation • ■ Retrieval latency and cost analysis
## Why this project?
Scientific literature is continuously growing, while finding the relevant evidence for a
specific question can require searching across many papers. PhysioEvidence explores how
modern retrieval techniques can be combined to make large collections of physiotherapy
research easier to search and understand while keeping generated answers connected to
their underlying sources.
