# Open questions

These questions do not block the local vertical slice and therefore have safe
defaults for now.

| Question | Safe default | Decision needed when |
| --- | --- | --- |
| Which object-storage provider should hold originals? | Local filesystem adapter | A deployment provider and official quota/terms links are supplied |
| Which public authentication provider should protect uploads? | Upload API is local-only by default; public deployment is blocked | Before exposing ingestion to an untrusted network |
| Which OpenAlex fields/source terms are approved? | OpenAlex is discovery-only and disabled by default | Metadata discovery is enabled |
| Should local embeddings replace Gemini in the production index? | Keep an embedding-model/versioned index boundary; benchmark before migration | A representative corpus and evaluation run exist |
| Which deployed PostgreSQL service supports pgvector? | PostgreSQL 16+ with pgvector, local or provider-confirmed | Cloud deployment is selected |
| Which UI hosting target is preferred? | The checked-in UI is a small local static app served by the API | A hosting target is selected |

No credentials, provider links, copyrighted papers, or benchmark claims are
required to complete the local implementation.
