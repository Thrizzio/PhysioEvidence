"""Unit and regression tests for Gemini LLM Provider."""

from unittest.mock import MagicMock
import pytest

from generation.llm import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMProvider,
    LLMResponse,
    sanitize_error_message,
)


def test_sanitize_error_message_redacts_api_key(monkeypatch):
    test_key = "AIzaSyFakeSecretKey1234567890abcdef"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)

    raw_error = f"Error connecting to https://generativelanguage.googleapis.com/v1beta/models?key={test_key} with key {test_key}"
    sanitized = sanitize_error_message(raw_error)

    assert test_key not in sanitized
    assert "key=REDACTED" in sanitized
    assert "REDACTED" in sanitized


def test_provider_initialization_default_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)

    provider = GeminiLLMProvider()
    assert provider.model_name == DEFAULT_GEMINI_MODEL
    assert provider.model_name == "gemini-3.5-flash-lite"


def test_provider_initialization_from_env_model(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

    provider = GeminiLLMProvider()
    assert provider.model_name == "gemini-3.1-pro-preview"


def test_provider_initialization_custom_model_arg(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    monkeypatch.setenv("GEMINI_MODEL", "ignored-env-model")

    provider = GeminiLLMProvider(model_name="explicit-model-name")
    assert provider.model_name == "explicit-model-name"


def test_provider_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        GeminiLLMProvider()

    monkeypatch.setenv("GEMINI_API_KEY", "   ")
    with pytest.raises(ValueError, match="GEMINI_API_KEY environment variable is required"):
        GeminiLLMProvider()


def test_generate_success_mocked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    provider = GeminiLLMProvider(model_name="gemini-3.8-flash")

    mock_response = MagicMock()
    mock_response.text = "Clinical trial results demonstrate significant pain reduction [Source 1]."
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 150
    mock_usage.candidates_token_count = 45
    mock_usage.total_token_count = 195
    mock_response.usage_metadata = mock_usage

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    monkeypatch.setattr("google.generativeai.GenerativeModel", MagicMock(return_value=mock_model))

    res = provider.generate("What is the effect?", system_prompt="You are a physiotherapy assistant.")

    assert isinstance(res, LLMResponse)
    assert res.content == "Clinical trial results demonstrate significant pain reduction [Source 1]."
    assert res.model_name == "gemini-3.8-flash"
    assert res.prompt_tokens == 150
    assert res.completion_tokens == 45
    assert res.total_tokens == 195


def test_generate_not_found_error_handling(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    provider = GeminiLLMProvider(model_name="gemini-1.5-flash")

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception(
        "404 models/gemini-1.5-flash is not found for API version v1beta, or is not supported for generateContent."
    )
    monkeypatch.setattr("google.generativeai.GenerativeModel", MagicMock(return_value=mock_model))

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("Test prompt")

    err_msg = str(exc_info.value)
    assert "gemini-1.5-flash" in err_msg
    assert "was not found or is no longer supported" in err_msg
    assert DEFAULT_GEMINI_MODEL in err_msg


def test_generate_quota_exceeded_error_handling(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "dummy_key_12345")
    provider = GeminiLLMProvider(model_name="gemini-3.1-pro-preview")

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception(
        "429 ResourceExhausted: Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 0, model: gemini-3.1-pro"
    )
    monkeypatch.setattr("google.generativeai.GenerativeModel", MagicMock(return_value=mock_model))

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate("Test prompt")

    err_msg = str(exc_info.value)
    assert "quota exceeded" in err_msg.lower()
    assert "free tier" in err_msg.lower()
