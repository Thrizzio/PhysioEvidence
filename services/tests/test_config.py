"""Unit tests for centralized configuration loading, credential masking, and validation."""

import pytest
from services.config import (
    AppConfig,
    ConfigurationError,
    DatabaseConfig,
    EmbeddingConfig,
    IngestionConfig,
    LLMConfig,
    EXPECTED_EMBEDDING_DIMENSION,
    mask_connection_string,
)


def test_mask_connection_string_uri():
    raw_uri = "postgresql://myuser:super_secret_password@ep-sample-pooler.neon.tech/physioevidence?sslmode=require"
    masked = mask_connection_string(raw_uri)
    assert "super_secret_password" not in masked
    assert "myuser:****@ep-sample-pooler.neon.tech/physioevidence" in masked
    assert "sslmode=require" in masked


def test_mask_connection_string_key_value():
    raw = "host=localhost port=5432 user=postgres password=mysecretpass dbname=test"
    masked = mask_connection_string(raw)
    assert "mysecretpass" not in masked
    assert "password=****" in masked


def test_mask_connection_string_empty():
    assert mask_connection_string("") == ""


def test_database_config_prefers_database_url():
    db = DatabaseConfig(
        database_url="postgresql://clouduser:cloudpass@neon.tech:5432/neondb",
        host="localhost",
        port=5432,
        db_name="localdb",
        user="localuser",
    )
    assert db.get_connection_string() == "postgresql://clouduser:cloudpass@neon.tech:5432/neondb"
    assert "cloudpass" not in db.get_masked_connection_string()


def test_database_config_fallback_formatting():
    db = DatabaseConfig(
        database_url=None,
        host="db.example.com",
        port=5433,
        db_name="test_physio",
        user="doctor",
        password="secretpassword",
    )
    assert db.get_connection_string() == "postgresql://doctor:secretpassword@db.example.com:5433/test_physio"
    assert "secretpassword" not in db.get_masked_connection_string()


def test_database_config_missing_parameters_raises():
    db = DatabaseConfig(
        database_url=None,
        host="",
        port=5432,
        db_name="test_physio",
        user="doctor",
    )
    with pytest.raises(ConfigurationError, match="Database host is missing"):
        db.validate_connection_parameters()


def test_embedding_config_dimensions_enforcement():
    # 768 is valid
    cfg = EmbeddingConfig(dimensions=768)
    assert cfg.dimensions == 768

    # Non-768 is strictly rejected
    with pytest.raises(ConfigurationError, match="Invalid embedding dimensions 512"):
        EmbeddingConfig(dimensions=512)

    with pytest.raises(ConfigurationError, match="Invalid embedding dimensions 1536"):
        EmbeddingConfig(dimensions=1536)


def test_embedding_config_provider_validation():
    # Valid providers
    assert EmbeddingConfig(provider="ollama").provider == "ollama"
    assert EmbeddingConfig(provider="OLLAMA").provider == "ollama"
    assert EmbeddingConfig(provider="gemini").provider == "gemini"
    assert EmbeddingConfig(provider="local").provider == "local"

    # Unsupported provider
    with pytest.raises(ConfigurationError, match="Unsupported EMBEDDING_PROVIDER 'unsupported'"):
        EmbeddingConfig(provider="unsupported")


def test_embedding_config_gemini_requires_api_key():
    cfg = EmbeddingConfig(provider="gemini", gemini_api_key=None)
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
        cfg.validate_for_provider()


def test_embedding_config_ollama_requires_base_url():
    cfg = EmbeddingConfig(provider="ollama", ollama_base_url="")
    with pytest.raises(ConfigurationError, match="OLLAMA_BASE_URL is required"):
        cfg.validate_for_provider()


def test_llm_config_requires_api_key_for_generation():
    cfg = LLMConfig(gemini_api_key=None)
    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is required"):
        cfg.validate_for_generation()

    valid_cfg = LLMConfig(gemini_api_key="valid-dummy-key")
    # Should not raise
    valid_cfg.validate_for_generation()


def test_llm_config_model_configuration():
    default_cfg = LLMConfig(gemini_api_key="valid-key")
    assert default_cfg.gemini_model == "gemini-3.5-flash-lite"

    custom_cfg = LLMConfig(gemini_api_key="valid-key", gemini_model="gemini-3.1-pro-preview")
    assert custom_cfg.gemini_model == "gemini-3.1-pro-preview"



def test_ingestion_config_file_size_validation():
    with pytest.raises(ConfigurationError, match="INGESTION_MAX_FILE_SIZE_BYTES must be positive"):
        IngestionConfig(max_file_size_bytes=0)

    with pytest.raises(ConfigurationError, match="INGESTION_MAX_FILE_SIZE_BYTES must be positive"):
        IngestionConfig(max_file_size_bytes=-100)


def test_ingestion_config_admin_token_validation():
    cfg = IngestionConfig(admin_token=None)
    with pytest.raises(ConfigurationError, match="INGESTION_ADMIN_TOKEN is not configured"):
        cfg.validate_for_admin_endpoint()


def test_app_config_from_env_with_monkeypatch(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db.test/physio")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "768")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "embeddinggemma-2:270m")

    app_cfg = AppConfig.from_env()
    assert app_cfg.database.database_url == "postgresql://u:p@db.test/physio"
    assert app_cfg.embedding.provider == "ollama"
    assert app_cfg.embedding.dimensions == 768
    assert app_cfg.embedding.ollama_base_url == "http://localhost:11434"
    assert app_cfg.embedding.ollama_embedding_model == "embeddinggemma-2:270m"
    # Startup validation succeeds
    app_cfg.validate_startup(service="ingestion_service")
