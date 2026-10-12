# Architecture Decision Records

ADRs document the reasoning and validation state of architectural decisions.
`Proposed` means the choice is implemented or staged but still requires
deployment/corpus evidence. `Accepted` is reserved for decisions validated in
the target environment. Existing ADRs 0001–0003 are retained as historical
records and are indexed below.

## Index

| ADR | Topic | Status |
| --- | --- | --- |
| [001](0001-source-agnostic-ingestion.md) | Source-agnostic ingestion and optional OpenAlex | Proposed |
| [002](0002-separate-ingestion-service.md) | Separate ingestion service boundary/API/job contract | Proposed |
| [003](0003-pdf-storage-and-postgres.md) | PDF originals versus PostgreSQL metadata/chunks/vectors | Proposed |
| [004](0004-postgresql-pgvector.md) | PostgreSQL + pgvector and scaling limits | Proposed |
| [005](0005-recursive-section-aware-chunking.md) | Recursive section-aware chunking | Accepted |
| [006](0006-chunk-context-and-versioning.md) | Chunk size, overlap, parent context, versioning | Proposed |
| [007](0007-local-embedding-model.md) | Local embedding model and compatibility | Proposed |
| [008](0008-dense-retrieval-baseline.md) | Dense retrieval baseline and index | Accepted |
| [009](0009-lexical-search.md) | PostgreSQL full-text ranking versus BM25 | Proposed |
| [010](0010-hybrid-rrf.md) | Hybrid fusion with RRF | Accepted |
| [011](0011-multi-query-retrieval.md) | Multi-query limits and fallback | Proposed |
| [012](0012-reranking.md) | Reranker activation criteria | Proposed |
| [013](0013-bibliographic-metadata.md) | Bibliographic metadata and filters | Proposed |
| [014](0014-grounded-generation-and-citations.md) | LLM context, citations, insufficient evidence | Accepted |
| [015](0015-langsmith-observability.md) | LangSmith tracing, redaction, opt-out, evaluation | Proposed |
| [016](0016-storage-quotas-and-licensing.md) | Quotas, deduplication, retention, licensing | Proposed |
| [017](0017-evaluation-and-reproducibility.md) | Evaluation dataset and metrics | Proposed |
| [018](0018-upload-authentication.md) | Upload authentication and authorization | Proposed |
| [019](0019-configuration-migrations-local-development.md) | Configuration, secrets, migrations, local development | Proposed |
| [020](0020-ui-structure.md) | UI and source/evidence presentation | Proposed |

Historical detailed records: [legacy storage ADR](0001-postgresql-pgvector-storage-architecture.md),
[legacy hybrid ADR](0002-hybrid-retrieval-with-reciprocal-rank-fusion.md), and
[legacy generation ADR](0003-grounded-generation-and-citation-architecture.md).
