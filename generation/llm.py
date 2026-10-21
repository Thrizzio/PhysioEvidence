import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from pydantic import BaseModel, Field
import google.generativeai as genai

logger = logging.getLogger(__name__)

import re

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"


def sanitize_error_message(error: Exception | str) -> str:
    """Sanitize error messages to ensure API keys and credentials are never exposed."""
    msg = str(error)
    # Redact common Google API key patterns and URL parameters
    msg = re.sub(r"key=[A-Za-z0-9_-]+", "key=REDACTED", msg)
    msg = re.sub(r"AIza[0-9A-Za-z-_]{35}", "AIza...REDACTED", msg)
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key and len(api_key.strip()) > 5:
        msg = msg.replace(api_key.strip(), "REDACTED")
    return msg


class LLMResponse(BaseModel):
    """
    Standardized response from an LLM call.
    """
    content: str
    model_name: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class LLMProvider(ABC):
    """
    Abstract interface for Language Model completions.
    """

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """
        Generates completion given a user prompt and optional system prompt.
        """
        pass


class GeminiLLMProvider(LLMProvider):
    """
    Production LLM provider using Google Gemini API.
    """

    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.2):
        self.model_name = model_name or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
        self.temperature = temperature

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        try:
            genai.configure(api_key=api_key.strip())
        except Exception as e:
            sanitized = sanitize_error_message(e)
            logger.error(f"Failed to configure Gemini API: {sanitized}")
            raise RuntimeError(f"Failed to configure Gemini API: {sanitized}") from None

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=system_prompt,
                generation_config={"temperature": self.temperature}
            )
            response = model.generate_content(prompt)
            text_content = response.text if response.text else ""

            # Extract token usage if returned by the API
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None) if hasattr(response, "usage_metadata") else None
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None) if hasattr(response, "usage_metadata") else None
            total_tokens = getattr(response.usage_metadata, "total_token_count", None) if hasattr(response, "usage_metadata") else None

            return LLMResponse(
                content=text_content,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens
            )
        except Exception as e:
            sanitized = sanitize_error_message(e)
            error_str = str(e).lower()
            if "404" in error_str and ("not found" in error_str or "no longer available" in error_str):
                clarification = (
                    f"Gemini model '{self.model_name}' was not found or is no longer supported for generateContent. "
                    f"Verify GEMINI_MODEL in your environment (recommended: '{DEFAULT_GEMINI_MODEL}')."
                )
            elif "429" in error_str or "quota" in error_str or "resourceexhausted" in error_str:
                if self.model_name == DEFAULT_GEMINI_MODEL:
                    clarification = (
                        f"Gemini API quota exceeded for model '{self.model_name}'. "
                        "Free-tier rate limits may be temporarily exhausted. Please retry shortly or verify project quota limits."
                    )
                else:
                    clarification = (
                        f"Gemini API quota exceeded for model '{self.model_name}'. "
                        "On the free tier, Pro models have a rate limit of 0; ensure you are using a supported tier "
                        f"or switch GEMINI_MODEL to '{DEFAULT_GEMINI_MODEL}'."
                    )
            elif "401" in error_str or "403" in error_str or "unauthenticated" in error_str:
                clarification = "Gemini API authentication failed. Verify that GEMINI_API_KEY is valid."
            else:
                clarification = sanitized

            logger.error(f"Gemini LLM generation failed for model '{self.model_name}': {sanitized}")
            raise RuntimeError(f"LLM generation error: {clarification}") from None


class MockLLMProvider(LLMProvider):
    """
    Deterministic mock provider for unit testing and offline verification.
    """

    def __init__(self, response_generator: Optional[Callable[[str], str]] = None):
        self.response_generator = response_generator

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        if self.response_generator:
            content = self.response_generator(prompt)
        else:
            content = "Based on the provided research [Source 1], the clinical intervention is effective."

        return LLMResponse(
            content=content,
            model_name="mock-llm-v1",
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130
        )
