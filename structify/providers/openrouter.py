"""OpenRouter provider for structify.

OpenRouter provides access to multiple LLM providers through a unified
OpenAI-compatible API, including Claude, GPT-4, Llama, Mistral, and more.
"""

import base64
import os
import time
from pathlib import Path
from typing import Any

from structify.providers.base import BaseLLMProvider
from structify.core.config import Config
from structify.core.exceptions import (
    ProviderError,
    ConfigurationError,
)
from structify.utils.logging import get_logger

logger = get_logger("openrouter")


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider using OpenAI-compatible interface.

    OpenRouter provides access to 100+ models from various providers:
    - Anthropic (Claude)
    - OpenAI (GPT-4, GPT-4o)
    - Google (Gemini)
    - Meta (Llama)
    - Mistral
    - And many more

    Handles all communication with the OpenRouter API including:
    - PDF processing via base64 encoding (for vision-capable models)
    - Content generation
    - Retry logic with exponential backoff
    - Rate limit handling
    """

    DEFAULT_MODEL = "anthropic/claude-sonnet-4"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 5,
        retry_delay: int = 60,
        between_calls_delay: int = 3,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        site_url: str | None = None,
        site_name: str | None = None,
    ):
        """
        Initialize the OpenRouter provider.

        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model name (defaults to anthropic/claude-sonnet-4)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            between_calls_delay: Delay between normal API calls
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
            site_url: Optional URL for OpenRouter rankings/analytics
            site_name: Optional name for OpenRouter rankings/analytics
        """
        if api_key is None:
            api_key = Config.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")

        if model is None:
            model = Config.get("default_model") or self.DEFAULT_MODEL

        super().__init__(
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            between_calls_delay=between_calls_delay,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        self.site_url = site_url or os.getenv("OPENROUTER_SITE_URL", "")
        self.site_name = site_name or os.getenv("OPENROUTER_SITE_NAME", "pdf-structify")
        self._client = None

    def initialize(self) -> None:
        """Initialize the OpenRouter API client."""
        if not self.api_key:
            raise ConfigurationError(
                "OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ConfigurationError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self._client = OpenAI(
            base_url=self.BASE_URL,
            api_key=self.api_key,
            default_headers={
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
        )
        self._is_initialized = True
        logger.info(f"Initialized OpenRouter provider with model: {self.model}")

    def _is_vision_model(self) -> bool:
        """Check if the current model supports vision/images."""
        vision_models = [
            "claude",
            "gpt-4o",
            "gpt-4-vision",
            "gemini",
            "llava",
            "qwen-vl",
            "pixtral",
        ]
        model_lower = self.model.lower()
        return any(vm in model_lower for vm in vision_models)

    def upload_file(self, file_path: str, mime_type: str = "application/pdf") -> Any:
        """
        Prepare a file for OpenRouter by encoding it as base64.

        For vision-capable models, files are sent inline as base64-encoded
        content. For text-only models, PDF text will need to be extracted first.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            Dict containing base64-encoded file data and metadata
        """
        self.ensure_initialized()

        logger.debug(f"Preparing file for OpenRouter: {file_path}")

        try:
            path = Path(file_path)
            if not path.exists():
                raise ProviderError(f"File not found: {file_path}")

            file_bytes = path.read_bytes()
            base64_data = base64.standard_b64encode(file_bytes).decode("utf-8")

            # For PDF files with vision models, use image_url format
            if mime_type == "application/pdf" and self._is_vision_model():
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}",
                    },
                }

            # For images
            if mime_type.startswith("image/"):
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_data}",
                    },
                }

            # For non-vision models or unsupported types, return raw for text extraction
            return {
                "_raw_bytes": file_bytes,
                "_file_path": str(path),
                "_mime_type": mime_type,
            }
        except ProviderError:
            raise
        except Exception as e:
            raise ProviderError(f"Failed to prepare file: {e}") from e

    def generate(
        self,
        prompt: str,
        file_ref: Any | None = None,
    ) -> str:
        """
        Generate a response from OpenRouter.

        Args:
            prompt: The prompt to send
            file_ref: Optional file reference from upload_file

        Returns:
            The generated text response
        """
        self.ensure_initialized()

        # Build content
        content = []
        if file_ref is not None:
            if "_raw_bytes" in file_ref:
                # Non-vision file - extract text if PDF
                if file_ref["_mime_type"] == "application/pdf":
                    try:
                        from pypdf import PdfReader
                        import io

                        reader = PdfReader(io.BytesIO(file_ref["_raw_bytes"]))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                        content.append({"type": "text", "text": f"[PDF Content]:\n{text}\n\n"})
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF text: {e}")
                        content.append({
                            "type": "text",
                            "text": f"[Unable to process PDF: {file_ref['_file_path']}]"
                        })
            else:
                content.append(file_ref)

        content.append({"type": "text", "text": prompt})

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": content}],
                )

                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message.content:
                        return message.content

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from OpenRouter API")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Rate limit handling
                if "429" in str(e) or "rate" in error_str or "quota" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Timeout handling
                if "timeout" in error_str or "timed out" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(
                        f"Timeout on attempt {attempt}/{self.max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                # Server errors
                if "500" in str(e) or "502" in str(e) or "503" in str(e):
                    wait_time = self.retry_delay * attempt
                    logger.warning(f"Server error, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"OpenRouter API error: {e}")
                raise ProviderError(f"OpenRouter API error: {e}") from e

        if last_exception:
            raise ProviderError(
                f"All {self.max_retries} attempts failed"
            ) from last_exception

        raise ProviderError("Generation failed with unknown error")

    def generate_with_file(
        self,
        prompt: str,
        file_path: str,
        mime_type: str = "application/pdf",
    ) -> str:
        """
        Prepare a file and generate a response in one call.

        Args:
            prompt: The prompt to send
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            The generated text response
        """
        file_ref = self.upload_file(file_path, mime_type)
        time.sleep(self.between_calls_delay)
        return self.generate(prompt, file_ref)

    def generate_with_files(
        self,
        prompt: str,
        file_refs: list[Any],
    ) -> str:
        """
        Generate a response with multiple files in one call.

        Args:
            prompt: The prompt to send
            file_refs: List of file references from upload_file()

        Returns:
            The generated text response
        """
        self.ensure_initialized()

        # Build content: process all files + prompt
        content = []
        for file_ref in file_refs:
            if "_raw_bytes" in file_ref:
                if file_ref["_mime_type"] == "application/pdf":
                    try:
                        from pypdf import PdfReader
                        import io

                        reader = PdfReader(io.BytesIO(file_ref["_raw_bytes"]))
                        text = "\n".join(page.extract_text() or "" for page in reader.pages)
                        content.append({"type": "text", "text": f"[PDF Content]:\n{text}\n\n"})
                    except Exception as e:
                        logger.warning(f"Failed to extract PDF text: {e}")
            else:
                content.append(file_ref)

        content.append({"type": "text", "text": prompt})

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": content}],
                )

                if response.choices and len(response.choices) > 0:
                    message = response.choices[0].message
                    if message.content:
                        return message.content

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from OpenRouter API")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                if "429" in str(e) or "rate" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if "timeout" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(f"Timeout on attempt {attempt}/{self.max_retries}...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"OpenRouter API error: {e}")
                raise ProviderError(f"OpenRouter API error: {e}") from e

        if last_exception:
            raise ProviderError(f"All {self.max_retries} attempts failed") from last_exception
        raise ProviderError("Generation failed with unknown error")

    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens in a text string.

        Note: OpenRouter doesn't have a direct token counting endpoint.
        Uses tiktoken for OpenAI models, rough estimation for others.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (estimated)
        """
        try:
            import tiktoken

            # Use cl100k_base as a reasonable default for most models
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            pass

        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def list_models(self) -> list[dict]:
        """
        List available models from OpenRouter.

        Returns:
            List of model information dictionaries
        """
        self.ensure_initialized()

        try:
            import requests

            response = requests.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []
