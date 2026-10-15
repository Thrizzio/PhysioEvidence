"""Centralized configuration loading and startup validation for PhysioEvidence.

Validates environment variables at application startup and provides clear,
actionable diagnostic errors for missing or invalid settings.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# Canonical vector dimension expected by PostgreSQL / pgvector schema
EXPECTED_EMBEDDING_DIMENSION = 768


class ConfigurationError(ValueError):
    """Raised when environment configuration is missing, invalid, or inconsistent."""
    pass


class _ConfigBase(BaseModel):
    """Base model that translates Pydantic validation errors into clean ConfigurationErrors."""

    def __init__(self, **data):
        try:
            super().__init__(**data)
        except ValidationError as e:
            messages = [err["msg"].removeprefix("Value error, ") for err in e.errors()]
            raise ConfigurationError("; ".join(messages)) from e


def mask_connection_string(conn_str: str) -> str:
    """Mask credentials in a PostgreSQL connection string for safe logging.

    Supports both URI (postgresql://user:pass@host/db) and keyword-value formats.
    Never exposes raw passwords or tokens.
    """
    if not conn_str:
        return ""
    try:
        parts = urlsplit(conn_str)
        if parts.password:
            # Replace password with asterisks
            masked_netloc = parts.netloc.replace(f":{parts.password}@", ":****@")
            return urlunsplit((parts.scheme, masked_netloc, parts.path, parts.query, parts.fragment))
        if "password=" in conn_str:
            return re.sub(r"password=\S+", "password=****", conn_str)
        return conn_str
    except Exception:
        return "postgresql://****"


class DatabaseConfig(_ConfigBase):
    """PostgreSQL and pgvector connection configuration."""
    database_url: Optional[str] = None
    host: str = "localhost"
    port: int = 5432
    db_name: str = "physioevidence"
    user: str = "postgres"
    password: str = ""

    def get_connection_string(self) -> str:
        """Return the active PostgreSQL connection string.

        Prefers DATABASE_URL (e.g. for Neon or cloud deployments) if set,
        falling back to discrete parameters for local development.
        """
        if self.database_url:
            return self.database_url
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.db_name}"

    def get_masked_connection_string(self) -> str:
        """Return connection string with passwords redacted."""
        return mask_connection_string(self.get_connection_string())

    def validate_connection_parameters(self) -> None:
        """Validate that minimum connection parameters are present."""
        if not self.database_url:
            if not self.host:
                raise ConfigurationError("Database host is missing. Set DATABASE_URL or POSTGRES_HOST.")
            if not self.db_name:
                raise ConfigurationError("Database name is missing. Set DATABASE_URL or POSTGRES_DB.")
            if not self.user:
                raise ConfigurationError("Database user is missing. Set DATABASE_URL or POSTGRES_USER.")


class EmbeddingConfig(_ConfigBase):
    """Embedding model and provider configuration."""
    provider: str = Field(default="ollama")
    dimensions: int = Field(default=EXPECTED_EMBEDDING_DIMENSION)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_embedding_model: str = Field(default="embeddinggemma-2:270m")
    gemini_embedding_model: str = Field(default="models/text-embedding-004")
    gemini_api_key: Optional[str] = None
    local_embedding_model: str = Field(default="google/embeddinggemma-300m")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        normalized = v.strip().lower()
        allowed = {"ollama", "local", "gemini"}
        if normalized not in allowed:
            raise ConfigurationError(
                f"Unsupported EMBEDDING_PROVIDER '{v}'. Must be one of: {sorted(allowed)}."
            )
        return normalized

    @field_validator("dimensions")
    @classmethod
    def validate_dimensions(cls, v: int) -> int:
        if v != EXPECTED_EMBEDDING_DIMENSION:
            raise ConfigurationError(
                f"Invalid embedding dimensions {v}. The pgvector schema requires exactly "
                f"{EXPECTED_EMBEDDING_DIMENSION} dimensions (vector({EXPECTED_EMBEDDING_DIMENSION}))."
            )
        return v

    def validate_for_provider(self) -> None:
        """Verify that provider-specific required variables are present."""
        if self.provider == "gemini":
            if not self.gemini_api_key or not self.gemini_api_key.strip():
                raise ConfigurationError(
                    "GEMINI_API_KEY is required when EMBEDDING_PROVIDER is 'gemini'. "
                    "Set GEMINI_API_KEY in your environment or switch EMBEDDING_PROVIDER to 'ollama'."
                )
        elif self.provider == "ollama":
            if not self.ollama_base_url or not self.ollama_base_url.strip():
                raise ConfigurationError(
                    "OLLAMA_BASE_URL is required when EMBEDDING_PROVIDER is 'ollama'. "
                    "Set OLLAMA_BASE_URL (default: 'http://localhost:11434')."
                )
            if not self.ollama_embedding_model or not self.ollama_embedding_model.strip():
                raise ConfigurationError(
                    "OLLAMA_EMBEDDING_MODEL is required when EMBEDDING_PROVIDER is 'ollama'. "
                    "Set OLLAMA_EMBEDDING_MODEL (default: 'embeddinggemma-2:270m')."
                )


class LLMConfig(_ConfigBase):
    """Large language model generation and synthesis configuration."""
    gemini_api_key: Optional[str] = None
    gemini_model: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)

    def validate_for_generation(self) -> None:
        """Ensure Gemini API key is configured before attempting generation."""
        if not self.gemini_api_key or not self.gemini_api_key.strip():
            raise ConfigurationError(
                "GEMINI_API_KEY is required for evidence synthesis and answer generation. "
                "Set GEMINI_API_KEY in your environment or .env file."
            )


class IngestionConfig(_ConfigBase):
    """Document ingestion service configuration."""
    storage_root: str = Field(default="./data/originals")
    max_file_size_bytes: int = Field(default=50 * 1024 * 1024)
    admin_token: Optional[str] = None
    quota_bytes: Optional[int] = None

    @field_validator("max_file_size_bytes")
    @classmethod
    def validate_file_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigurationError(f"INGESTION_MAX_FILE_SIZE_BYTES must be positive, got {v}.")
        return v

    @field_validator("quota_bytes")
    @classmethod
    def validate_quota(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ConfigurationError(f"INGESTION_QUOTA_BYTES must be positive if set, got {v}.")
        return v


    def validate_for_admin_endpoint(self) -> None:
        """Validate admin token is configured when running protected ingestion."""
        if not self.admin_token or not self.admin_token.strip():
            raise ConfigurationError(
                "INGESTION_ADMIN_TOKEN is not configured. Admin endpoints require an authorization token."
            )


class CORSConfig(_ConfigBase):
    """Allowed CORS origins for web clients."""
    research_ui_origins: list[str] = Field(default_factory=list)
    ingestion_ui_origins: list[str] = Field(default_factory=list)


class ObservabilityConfig(_ConfigBase):
    """LangSmith observability and tracing settings."""
    tracing: bool = False
    api_key: Optional[str] = None
    project: str = "physioevidence"
    endpoint: str = "https://api.smith.langchain.com"


class AppConfig(_ConfigBase):
    """Root configuration aggregating all subsystem settings."""
    database: DatabaseConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    ingestion: IngestionConfig
    cors: CORSConfig
    observability: ObservabilityConfig

    @classmethod
    def from_env(cls) -> AppConfig:
        """Construct validated AppConfig from the environment.

        Reads .env if present without modifying it.
        """
        load_dotenv()

        # Database
        port_raw = os.getenv("POSTGRES_PORT", "5432")
        try:
            port = int(port_raw)
        except ValueError:
            raise ConfigurationError(f"Invalid POSTGRES_PORT '{port_raw}': must be an integer.")

        database = DatabaseConfig(
            database_url=os.getenv("DATABASE_URL") or None,
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=port,
            db_name=os.getenv("POSTGRES_DB", "physioevidence"),
            user=os.getenv("POSTGRES_USER", "postgres"),
            password=os.getenv("POSTGRES_PASSWORD", ""),
        )

        # Embedding
        dim_raw = (
            os.getenv("EMBEDDING_DIMENSIONS")
            or os.getenv("OLLAMA_EMBEDDING_DIMENSIONS")
            or os.getenv("LOCAL_EMBEDDING_DIMENSIONS")
            or str(EXPECTED_EMBEDDING_DIMENSION)
        )
        try:
            dimensions = int(dim_raw)
        except ValueError:
            raise ConfigurationError(f"Invalid EMBEDDING_DIMENSIONS '{dim_raw}': must be an integer.")

        provider = os.getenv("EMBEDDING_PROVIDER", "ollama")

        embedding = EmbeddingConfig(
            provider=provider,
            dimensions=dimensions,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma-2:270m"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m"),
        )

        # LLM
        llm = LLMConfig(
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        )

        # Ingestion
        max_size_raw = os.getenv("INGESTION_MAX_FILE_SIZE_BYTES", str(50 * 1024 * 1024))
        try:
            max_size = int(max_size_raw)
        except ValueError:
            raise ConfigurationError(f"Invalid INGESTION_MAX_FILE_SIZE_BYTES '{max_size_raw}'.")

        quota_raw = os.getenv("INGESTION_QUOTA_BYTES")
        quota = None
        if quota_raw and quota_raw.strip():
            try:
                quota = int(quota_raw)
            except ValueError:
                raise ConfigurationError(f"Invalid INGESTION_QUOTA_BYTES '{quota_raw}'.")

        ingestion = IngestionConfig(
            storage_root=os.getenv("INGESTION_STORAGE_ROOT", "./data/originals"),
            max_file_size_bytes=max_size,
            admin_token=os.getenv("INGESTION_ADMIN_TOKEN") or None,
            quota_bytes=quota,
        )

        # CORS
        def _parse_origins(val: str, defaults: list[str]) -> list[str]:
            configured = [o.strip() for o in val.split(",") if o.strip()]
            base = configured if configured else defaults
            expanded = list(base)
            for origin in base:
                if origin.startswith("http://localhost:"):
                    port = origin.split(":")[-1]
                    alt = f"http://127.0.0.1:{port}"
                    if alt not in expanded:
                        expanded.append(alt)
                elif origin.startswith("http://127.0.0.1:"):
                    port = origin.split(":")[-1]
                    alt = f"http://localhost:{port}"
                    if alt not in expanded:
                        expanded.append(alt)
            return expanded

        cors = CORSConfig(
            research_ui_origins=_parse_origins(
                os.getenv("RESEARCH_UI_ORIGIN", ""),
                ["http://localhost:8080", "http://127.0.0.1:8080"],
            ),
            ingestion_ui_origins=_parse_origins(
                os.getenv("INGESTION_UI_ORIGIN", ""),
                ["http://localhost:8081", "http://127.0.0.1:8081"],
            ),
        )

        # Observability
        observability = ObservabilityConfig(
            tracing=os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes"),
            api_key=os.getenv("LANGSMITH_API_KEY") or None,
            project=os.getenv("LANGSMITH_PROJECT", "physioevidence"),
            endpoint=os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"),
        )

        return cls(
            database=database,
            embedding=embedding,
            llm=llm,
            ingestion=ingestion,
            cors=cors,
            observability=observability,
        )

    def validate_startup(self, service: Optional[str] = None) -> None:
        """Validate configuration at startup with actionable error messages."""
        self.database.validate_connection_parameters()
        self.embedding.validate_for_provider()

        if service in ("research_api", "rag_cli"):
            self.llm.validate_for_generation()
        elif service == "ingestion_service":
            # If admin token is configured or required
            pass


_cached_config: Optional[AppConfig] = None


def get_app_config(reload: bool = False) -> AppConfig:
    """Return the cached application configuration or load it from the environment."""
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = AppConfig.from_env()
    return _cached_config
