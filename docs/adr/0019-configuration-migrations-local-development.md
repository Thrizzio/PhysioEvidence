# ADR-019: Configuration, secrets, migrations, and local development
- Status: Accepted
- Date: 2026-10-15
- Context: External services (PostgreSQL/pgvector, Ollama local embeddings, Google Gemini, LangSmith) require consistent, validated configuration across local development and cloud deployments (e.g. Neon). Silent fallback to remote APIs or loose environment variable lookups leads to undetected runtime misconfiguration or unexpected billing. In addition, logging connection strings risks credential leakage.
- Decision:
  1. Centralize environment configuration and validation in `services/config.py` using Pydantic models (`AppConfig`, `DatabaseConfig`, `EmbeddingConfig`, etc.).
  2. Enforce strict startup validation without silent fallbacks: if `EMBEDDING_PROVIDER=ollama`, Ollama settings are validated; if `EMBEDDING_PROVIDER=gemini`, `GEMINI_API_KEY` is strictly required. Unconfigured providers raise clear `ConfigurationError` exceptions.
  3. Enforce the verified 768 embedding dimensions (`vector(768)`) across local Ollama (`embeddinggemma-2:270m`) and remote configurations to match the PostgreSQL schema.
  4. Redact credentials from PostgreSQL connection strings in logs using `mask_connection_string`.
  5. Provide diagnostics for remote PostgreSQL connection timeouts (e.g. Neon TCP port 5432 firewall restrictions on local host networks).
  6. Maintain `.env.example` with safe placeholders and clean separation between local development defaults and production deployment settings; `.env` remains gitignored and must never be committed or overwritten.
- Alternatives considered:
  - Ad-hoc `os.getenv` calls scattered throughout codebase: Prone to silent defaults, inconsistent key names, and late-stage runtime crashes.
  - Automatic fallback from local Ollama to Gemini: Disallowed because it violates local privacy/cost expectations and silently masks local infrastructure issues.
  - Hardcoded connection strings or embedding dimensions: Inflexible across environments.
- Consequences and trade-offs: Users must configure services explicitly; missing variables fail fast at startup with actionable error messages.
- Security, privacy, and licensing: `.env` and secret keys are never committed; logs redact connection passwords; local embedding runs entirely on-premise.
- Performance/scaling implications: Local in-memory store keeps unit tests fast (offline in <10s); Neon / pgvector handles production-scale persistence.
- Validation/evidence: 76-test suite passing with 100% success; real Neon PostgreSQL and Ollama 768-dim vector insert/retrieval verified; credential masking unit tested.
