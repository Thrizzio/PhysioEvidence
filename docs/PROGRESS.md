# Progress

## Current session

### Completed

- Audited the repository, Git history, dependencies, tests, and existing
  document-processing/storage/retrieval/generation abstractions.
- Confirmed the repository was clean before implementation.
- Recorded the implementation plan and unresolved external decisions.
- Added content-hash document identity, deterministic chunk identity, page and
  processing metadata, a versioned PostgreSQL migration, and an optional local
  EmbeddingGemma provider.
- Added separate public research and private ingestion APIs, two separate static
  websites, admin-token protection, upload validation, per-file terminal job
  records, local original storage accounting, and deduplication.
- Added generated-citation validation and evidence excerpts/metadata for the
  research UI.

### Baseline verification

- `python3 --version` -> Python 3.14.4
- `pytest -q` -> not run: `pytest` is not installed in the current environment.
- `python --version` -> not available; use `python3` in this environment.
- `python3 -m compileall -q .` -> passed.

### Blockers and limitations

- PostgreSQL, pgvector, Gemini credentials, LangSmith credentials, and an
  approved object-storage provider are not available in this environment.
- The current environment cannot execute pytest because neither pytest nor pip
  is installed. The added ingestion and citation tests are checked in but not
  executed here.

### Next task

Install project dependencies in a clean environment, execute the full test
suite, then validate the FastAPI endpoints with injected in-memory components.

### Resume commands

```bash
python3 -m compileall .
python3 -m pytest -q
```
