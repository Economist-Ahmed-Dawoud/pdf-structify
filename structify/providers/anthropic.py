"""Anthropic Claude provider for structify."""

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

logger = get_logger("anthropic")


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude API provider.

    Handles all communication with the Anthropic API including:
    - PDF processing via base64 encoding (Claude supports PDF via vision)
    - Content generation
    - Retry logic with exponential backoff
    - Rate limit handling
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

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
    ):
        """
        Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Model name (defaults to claude-sonnet-4-20250514)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            between_calls_delay: Delay between normal API calls
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
        """
        if api_key is None:
            api_key = Config.get("anthropic_api_key") or os.getenv("ANTHROPIC_API_KEY")

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

        self._client = None

    def initialize(self) -> None:
        """Initialize the Anthropic API client."""
        if not self.api_key:
            raise ConfigurationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        try:
            import anthropic
        except ImportError:
            raise ConfigurationError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        self._client = anthropic.Anthropic(api_key=self.api_key)
        self._is_initialized = True
        logger.info(f"Initialized Anthropic provider with model: {self.model}")

    def upload_file(self, file_path: str, mime_type: str = "application/pdf") -> Any:
        """
        Prepare a file for Claude by encoding it as base64.

        Claude doesn't have a file upload API like Gemini. Instead, files are
        sent inline as base64-encoded content in the message.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            Dict containing base64-encoded file data and metadata
        """
        self.ensure_initialized()

        logger.debug(f"Preparing file for Claude: {file_path}")

        try:
            path = Path(file_path)
            if not path.exists():
                raise ProviderError(f"File not found: {file_path}")

            file_bytes = path.read_bytes()
            base64_data = base64.standard_b64encode(file_bytes).decode("utf-8")

            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": mime_type,
                    "data": base64_data,
                },
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
        Generate a response from Claude.

        Args:
            prompt: The prompt to send
            file_ref: Optional file reference from upload_file (base64 dict)

        Returns:
            The generated text response
        """
        self.ensure_initialized()

        # Build content blocks
        content = []
        if file_ref is not None:
            content.append(file_ref)
        content.append({"type": "text", "text": prompt})

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": content}],
                )

                # Extract text from response
                if response.content and len(response.content) > 0:
                    text_blocks = [
                        block.text for block in response.content if block.type == "text"
                    ]
                    if text_blocks:
                        return "\n".join(text_blocks)

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from Anthropic API")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Rate limit handling
                if "429" in str(e) or "rate" in error_str or "overloaded" in error_str:
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

                # Check for anthropic-specific errors
                if hasattr(e, "status_code"):
                    if e.status_code == 529:  # Overloaded
                        wait_time = self.retry_delay
                        logger.warning(f"API overloaded, waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                logger.error(f"Anthropic API error: {e}")
                raise ProviderError(f"Anthropic API error: {e}") from e

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

        # Build content: all files + prompt
        content = file_refs + [{"type": "text", "text": prompt}]

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": content}],
                )

                if response.content and len(response.content) > 0:
                    text_blocks = [
                        block.text for block in response.content if block.type == "text"
                    ]
                    if text_blocks:
                        return "\n".join(text_blocks)

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from Anthropic API")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                if "429" in str(e) or "rate" in error_str or "overloaded" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(f"Rate limit hit, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if "timeout" in error_str or "timed out" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(
                        f"Timeout on attempt {attempt}/{self.max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                logger.error(f"Anthropic API error: {e}")
                raise ProviderError(f"Anthropic API error: {e}") from e

        if last_exception:
            raise ProviderError(f"All {self.max_retries} attempts failed") from last_exception
        raise ProviderError("Generation failed with unknown error")

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
            result = self._client.count_tokens(text)
            return result
        except Exception as e:
            logger.warning(f"Failed to count tokens: {e}")
            # Rough estimate: ~4 characters per token for Claude
            return len(text) // 4
