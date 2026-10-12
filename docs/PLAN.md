# Implementation plan

## Verified baseline

The repository already contains a Python document-processing, storage, retrieval,
generation, evaluation, and optional tracing foundation. The current tests are
unit/component tests and do not require PostgreSQL, Gemini, or LangSmith.

## Delivery order

1. Make document identity, extraction status, chunk identity, page/section
   metadata, and processing versions explicit and deterministic.
2. Add a local-first embedding adapter. The selected local model is optional and
   is only considered usable after a local install and retrieval benchmark; the
   existing Gemini provider remains available as an explicitly configured remote
   option.
3. Add idempotent filesystem original storage and a separate ingestion service
   with a documented job contract, upload/status/list endpoints, and a CLI.
4. Harden retrieval and citation verification, then expose a small HTTP API and
   accessible research UI backed by the existing RAG abstractions.
5. Add reproducible evaluation output, optional LangSmith configuration, and
   complete ADRs/progress documentation.

## Working-slice acceptance checks

- A valid PDF can be extracted, sectioned, chunked, embedded with a test/local
  provider, and persisted to an in-memory store.
- A second submission of the same bytes is reported as a duplicate and does not
  create another document or embedding set.
- Invalid, empty/scanned, extraction, embedding, and storage failures produce a
  visible terminal failure state.
- Every displayed evidence source is constructed from an actual retrieved chunk
  and retains document/page/section identity.
- Dense and hybrid retrieval remain selectable and evaluable offline.
- Documentation clearly labels implemented behavior, deferred work, and external
  validation that has not been run.
