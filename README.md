# PhysioEvidence

PhysioEvidence is an evidence-grounded research assistant for physiotherapy
literature. It has two deliberately separate websites:

1. **Research website (`web/research/`)** — a minimal public question box. It
   calls the research API and displays a grounded answer, exact retrieved
   excerpts, and document/page/section metadata.
2. **Private ingestion website (`web/admin/`)** — a corpus-owner-only upload
   page. It calls the separate ingestion service and shows validation,
   deduplication, indexing state, errors, and storage usage.

The public site has no upload controls. The ingestion API requires
`INGESTION_ADMIN_TOKEN` before it will accept or list documents.

## Implemented architecture

```text
Research website                 Private ingestion website
       │                                  │
       ▼                                  ▼
services/research_api       services/ingestion_service
       │                                  │
       └──────────────┬───────────────────┘
                      ▼
       document_processing / storage / retrieval / generation
                      │
             PostgreSQL + pgvector
```

The ingestion flow is:

```text
PDF upload → size/type/parser validation → SHA-256 deduplication
→ PyMuPDF pages → section detection → deterministic recursive chunks
→ configured embeddings → PostgreSQL/pgvector persistence → terminal job state
```

The research flow is:

```text
Question → dense or hybrid RRF retrieval → bounded context
→ grounded LLM synthesis → verified source references
```

The application refuses to present a generated answer as verified when the LLM
returns a citation number not present in the retrieved context. Retrieved PDF
text is treated as untrusted data, not instructions.

## Repository layout

- `document_processing/` — PyMuPDF extraction, cleaning, section detection,
  deterministic section-aware chunking, and embedding providers.
- `storage/` — PostgreSQL/pgvector schema and store plus an in-memory store for
  offline tests. `storage/migrations/` contains forward migrations.
- `retrieval/` — dense retrieval, PostgreSQL/in-memory lexical retrieval, RRF
  hybrid retrieval, and optional reranking.
- `generation/` — context budgeting, grounded prompts, citation mapping, and
  LLM provider abstraction.
- `rag/` — end-to-end question orchestration and CLI.
- `services/research_api/` — public question API; no upload endpoints.
- `services/ingestion_service/` — private upload validation, job state,
  original-PDF storage, deduplication, and indexing API.
- `web/research/` — public research website.
- `web/admin/` — private admin website.
- `evaluation/` — retrieval metrics and a small synthetic benchmark corpus.
- `docs/adr/` — decision records and validation status.

## Local setup

Python 3.14 is the project target.

```bash
python3 -m venv .venv
. .venv/bin/activate                 # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
cp .env.example .env                # Windows: copy .env.example .env
```

Set actual local values in `.env`. Never commit `.env`, API keys, database
passwords, private PDFs, or copyrighted papers that you are not allowed to
redistribute.

Initialize a PostgreSQL 16+ database with pgvector:

```bash
python3 -m document_processing.cli --init-db
python3 -m document_processing.cli --migrate-db
```

For a database created before processing metadata was added, `--migrate-db`
applies the checked-in SQL in `storage/migrations/001_processing_metadata.sql`.

## Run both APIs

The APIs can be started independently, which is also how the two sites remain
separate in deployment:

```bash
uvicorn services.research_api.app:app --host 127.0.0.1 --port 8000
uvicorn services.ingestion_service.app:app --host 127.0.0.1 --port 8001
```

Configure these values in `.env` for the private service:

```text
INGESTION_ADMIN_TOKEN=use_a_long_local_secret
INGESTION_STORAGE_ROOT=./data/originals
RESEARCH_UI_ORIGIN=http://localhost:8080
INGESTION_UI_ORIGIN=http://localhost:8081
```

Serve the websites separately:

```bash
python3 -m http.server 8080 --directory web/research
python3 -m http.server 8081 --directory web/admin
```

The HTML files default to the API URLs above. For deployment, set the
`data-api-base` value in each site and configure exact CORS origins. Do not use
the admin site or upload API on a public network until authentication,
authorization, rate limits, TLS, and secret management are configured.

## Ingestion commands

Dry-run a PDF without embeddings:

```bash
python3 -m document_processing.cli --input data/sample_paper.pdf --output chunks.json --no-embeddings
```

Ingest a folder into PostgreSQL:

```bash
python3 -m document_processing.cli --input-dir data/papers --recursive --save-db
```

The standalone ingestion website uses the same canonical pipeline; it does not
duplicate extraction or chunking logic. Content-derived document IDs and
deterministic chunk IDs make repeated submission of identical bytes idempotent.

## Embeddings

Gemini embeddings remain an optional remote provider. The local-first provider
is optional because model weights are large:

```bash
python3 -m pip install -r requirements-local-embeddings.txt
```

Set `EMBEDDING_PROVIDER=local` to use `google/embeddinggemma-300m` through
sentence-transformers. The model card was checked for the model name, Gemma
license, 768-dimensional output, 2048-token input limit, and query/document
prompt behavior. Local CPU/RAM throughput and retrieval quality have not yet
been benchmarked on a representative PhysioEvidence corpus, so this is a
configured option rather than an improvement claim. Never mix dimensions or
model versions in one vector index.

## Query and retrieval CLI

```bash
python3 -m rag.cli --question "What quadriceps symmetry index is needed after ACL reconstruction?" --hybrid --rerank
python3 -m evaluation.run_benchmark
```

The checked-in evaluation corpus is synthetic and suitable for exercising
metrics and strategy comparison. It is not evidence from real papers and its
results must not be reported as clinical retrieval performance.

## Verification

```bash
python3 -m compileall -q .
python3 -m pytest -q
```

External PostgreSQL, pgvector, Gemini, LangSmith, and cloud-storage checks
require local credentials/services and are not silently represented as passed.
LangSmith is disabled by default and can be enabled with the documented
environment variables in `.env.example`.

## ADRs and project record

- [Implementation plan](docs/PLAN.md)
- [Open questions](docs/OPEN_QUESTIONS.md)
- [Progress log](docs/PROGRESS.md)
- [ADR index](docs/adr/README.md)
