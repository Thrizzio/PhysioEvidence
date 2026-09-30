import os
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
from pydantic import BaseModel, Field
import google.generativeai as genai

logger = logging.getLogger(__name__)

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

    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required.")
        genai.configure(api_key=api_key)

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
            logger.error(f"Gemini LLM generation failed: {e}")
            raise RuntimeError(f"LLM generation error: {e}") from e

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
