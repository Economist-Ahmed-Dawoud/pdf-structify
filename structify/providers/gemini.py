"""Google Gemini provider for structify using new google.genai SDK."""

import os
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from structify.providers.base import BaseLLMProvider
from structify.core.config import Config
from structify.core.exceptions import (
    ProviderError,
    RateLimitError,
    ConfigurationError,
)
from structify.utils.logging import get_logger
from structify.utils.retry import is_retryable_error, classify_error

logger = get_logger("gemini")


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider using the new google.genai SDK.

    Handles all communication with the Gemini API including:
    - File uploads
    - Content generation
    - Retry logic with exponential backoff
    - Rate limit handling
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 5,
        retry_delay: int = 60,
        between_calls_delay: int = 3,
        temperature: float = 0.1,
        max_output_tokens: int = 60000,
    ):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY env var)
            model: Model name (defaults to gemini-2.0-flash)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            between_calls_delay: Delay between normal API calls
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
        """
        # Get API key from config or environment
        if api_key is None:
            api_key = Config.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")

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

        self._client: genai.Client | None = None

    def initialize(self) -> None:
        """Initialize the Gemini API client."""
        if not self.api_key:
            raise ConfigurationError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Create client with API key
        self._client = genai.Client(api_key=self.api_key)

        self._is_initialized = True
        logger.info(f"Initialized Gemini provider with model: {self.model}")

    def upload_file(self, file_path: str, mime_type: str = "application/pdf") -> Any:
        """
        Upload a file to Gemini.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            Gemini file reference object
        """
        self.ensure_initialized()

        logger.debug(f"Uploading file: {file_path}")

        try:
            # Read file as bytes to avoid unicode filename issues
            path = Path(file_path)
            file_bytes = path.read_bytes()

            # Use the new SDK's file upload with bytes
            file_ref = self._client.files.upload(
                file=file_bytes,
                config=types.UploadFileConfig(
                    mime_type=mime_type,
                    display_name=path.name.encode('ascii', 'replace').decode('ascii')
                )
            )
            return file_ref
        except Exception as e:
            if is_retryable_error(e):
                error_class = classify_error(e)
                raise error_class(str(e)) from e
            raise ProviderError(f"Failed to upload file: {e}") from e

    def generate(
        self,
        prompt: str,
        file_ref: Any | None = None,
    ) -> str:
        """
        Generate a response from Gemini.

        Args:
            prompt: The prompt to send
            file_ref: Optional file reference from upload_file

        Returns:
            The generated text response
        """
        self.ensure_initialized()

        # Build content
        contents = [prompt]
        if file_ref is not None:
            contents = [file_ref, prompt]

        # Generation config
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )

        # Call with retry logic
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                # Check for valid response
                if not response.text:
                    logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        time.sleep(self.between_calls_delay)
                        continue
                    raise ProviderError("Empty response from Gemini API")

                return response.text

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Rate limit handling
                if "429" in str(e) or "quota" in error_str or "resource" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Timeout handling (504 Deadline Exceeded)
                if "504" in str(e) or "deadline" in error_str or "timeout" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(
                        f"Timeout (504) on attempt {attempt}/{self.max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                # Empty response handling
                if "response.text" in str(e) or "valid `Part`" in str(e):
                    logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                    if attempt < self.max_retries:
                        time.sleep(self.between_calls_delay)
                        continue
                    raise ProviderError("Empty response from Gemini API") from e

                # Other errors
                logger.error(f"Gemini API error: {e}")
                raise ProviderError(f"Gemini API error: {e}") from e

        # All retries exhausted
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
        Upload a file and generate a response in one call.

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

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        self.ensure_initialized()

        try:
            result = self._client.models.count_tokens(
                model=self.model,
                contents=text,
            )
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Rough estimate: ~4 characters per token
            return len(text) // 4
