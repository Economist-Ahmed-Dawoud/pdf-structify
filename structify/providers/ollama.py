"""Ollama provider for structify - Local LLM support.

Ollama allows running LLMs locally with support for many open-source models
including Llama, Mistral, Qwen, and vision models like LLaVA.
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

logger = get_logger("ollama")


class OllamaProvider(BaseLLMProvider):
    """
    Ollama provider for local LLM inference.

    Ollama provides easy access to local LLMs including:
    - Llama 3.1, Llama 3.2
    - Mistral, Mixtral
    - Qwen, Qwen2
    - LLaVA (vision)
    - And many more

    Handles all communication with the Ollama API including:
    - PDF processing via base64 encoding (for vision models) or text extraction
    - Content generation
    - Retry logic
    - Connection handling
    """

    DEFAULT_MODEL = "llama3.2"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 300,  # Local models can be slower
        max_retries: int = 3,
        retry_delay: int = 10,
        between_calls_delay: int = 1,
        temperature: float = 0.1,
        max_output_tokens: int = 8192,
        api_key: str | None = None,  # Not used but kept for interface compatibility
    ):
        """
        Initialize the Ollama provider.

        Args:
            base_url: Ollama server URL (defaults to OLLAMA_BASE_URL or localhost:11434)
            model: Model name (defaults to llama3.2)
            timeout: Request timeout in seconds (higher for local inference)
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            between_calls_delay: Delay between normal API calls
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens (num_predict in Ollama)
            api_key: Not used - Ollama doesn't require authentication
        """
        if base_url is None:
            base_url = Config.get("ollama_base_url") or os.getenv(
                "OLLAMA_BASE_URL", self.DEFAULT_BASE_URL
            )

        if model is None:
            model = Config.get("default_model") or self.DEFAULT_MODEL

        # Store base_url before calling super().__init__
        self.base_url = base_url.rstrip("/")

        super().__init__(
            api_key=None,  # Ollama doesn't use API keys
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
        """Initialize the Ollama client and verify connection."""
        try:
            import ollama
        except ImportError:
            raise ConfigurationError(
                "Ollama package not installed. Install with: pip install ollama"
            )

        try:
            # Create client
            self._client = ollama.Client(host=self.base_url)

            # Verify connection by listing models
            models = self._client.list()
            available_models = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
            logger.debug(f"Available Ollama models: {available_models}")

            # Check if requested model is available
            model_available = any(
                self.model in m or m.startswith(self.model.split(":")[0])
                for m in available_models
            )

            if not model_available and available_models:
                logger.warning(
                    f"Model '{self.model}' not found locally. "
                    f"Available: {available_models}. Ollama will attempt to pull it."
                )

        except Exception as e:
            raise ConfigurationError(
                f"Failed to connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: {e}"
            )

        self._is_initialized = True
        logger.info(f"Initialized Ollama provider with model: {self.model}")

    def _is_vision_model(self) -> bool:
        """Check if the current model supports vision/images."""
        vision_models = [
            "llava",
            "bakllava",
            "moondream",
            "llama3.2-vision",
            "minicpm-v",
        ]
        model_lower = self.model.lower()
        return any(vm in model_lower for vm in vision_models)

    def upload_file(self, file_path: str, mime_type: str = "application/pdf") -> Any:
        """
        Prepare a file for Ollama.

        For vision models, files are sent as base64-encoded images.
        For text models, PDF text is extracted.

        Args:
            file_path: Path to the file
            mime_type: MIME type of the file

        Returns:
            Dict containing file data for generation
        """
        self.ensure_initialized()

        logger.debug(f"Preparing file for Ollama: {file_path}")

        try:
            path = Path(file_path)
            if not path.exists():
                raise ProviderError(f"File not found: {file_path}")

            file_bytes = path.read_bytes()

            # For vision models and image files
            if self._is_vision_model() and (
                mime_type.startswith("image/") or mime_type == "application/pdf"
            ):
                base64_data = base64.standard_b64encode(file_bytes).decode("utf-8")
                return {
                    "_type": "image",
                    "_data": base64_data,
                    "_mime_type": mime_type,
                }

            # For PDF files with non-vision models, extract text
            if mime_type == "application/pdf":
                try:
                    from pypdf import PdfReader
                    import io

                    reader = PdfReader(io.BytesIO(file_bytes))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                    return {
                        "_type": "text",
                        "_content": text,
                        "_file_path": str(path),
                    }
                except Exception as e:
                    logger.warning(f"Failed to extract PDF text: {e}")
                    raise ProviderError(f"Cannot process PDF with non-vision model: {e}")

            # For other files, return raw for text extraction
            return {
                "_type": "raw",
                "_bytes": file_bytes,
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
        Generate a response from Ollama.

        Args:
            prompt: The prompt to send
            file_ref: Optional file reference from upload_file

        Returns:
            The generated text response
        """
        self.ensure_initialized()

        # Build the request
        images = []
        full_prompt = prompt

        if file_ref is not None:
            if file_ref.get("_type") == "image":
                # For vision models
                images.append(file_ref["_data"])
            elif file_ref.get("_type") == "text":
                # For extracted text
                full_prompt = f"[Document Content]:\n{file_ref['_content']}\n\n{prompt}"

        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                options = {
                    "temperature": self.temperature,
                    "num_predict": self.max_output_tokens,
                }

                if images:
                    response = self._client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        images=images,
                        options=options,
                    )
                else:
                    response = self._client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        options=options,
                    )

                if response and response.get("response"):
                    return response["response"]

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from Ollama")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                # Connection errors
                if "connection" in error_str or "refused" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(
                        f"Connection error on attempt {attempt}/{self.max_retries}, "
                        f"waiting {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue

                # Model loading
                if "loading" in error_str or "pulling" in error_str:
                    wait_time = self.retry_delay * 2
                    logger.info(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                # Timeout
                if "timeout" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(f"Timeout on attempt {attempt}/{self.max_retries}...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"Ollama error: {e}")
                raise ProviderError(f"Ollama error: {e}") from e

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

        # Combine all file content
        images = []
        text_parts = []

        for file_ref in file_refs:
            if file_ref.get("_type") == "image":
                images.append(file_ref["_data"])
            elif file_ref.get("_type") == "text":
                text_parts.append(f"[Document Content]:\n{file_ref['_content']}")

        full_prompt = "\n\n".join(text_parts + [prompt]) if text_parts else prompt

        last_exception = None
        for attempt in range(1, self.max_retries + 1):
            try:
                options = {
                    "temperature": self.temperature,
                    "num_predict": self.max_output_tokens,
                }

                if images:
                    response = self._client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        images=images,
                        options=options,
                    )
                else:
                    response = self._client.generate(
                        model=self.model,
                        prompt=full_prompt,
                        options=options,
                    )

                if response and response.get("response"):
                    return response["response"]

                logger.warning(f"Empty response on attempt {attempt}/{self.max_retries}")
                if attempt < self.max_retries:
                    time.sleep(self.between_calls_delay)
                    continue
                raise ProviderError("Empty response from Ollama")

            except Exception as e:
                last_exception = e
                error_str = str(e).lower()

                if "connection" in error_str:
                    wait_time = self.retry_delay
                    logger.warning(f"Connection error, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                if "timeout" in error_str:
                    wait_time = self.retry_delay * attempt
                    logger.warning(f"Timeout on attempt {attempt}...")
                    time.sleep(wait_time)
                    continue

                logger.error(f"Ollama error: {e}")
                raise ProviderError(f"Ollama error: {e}") from e

        if last_exception:
            raise ProviderError(f"All {self.max_retries} attempts failed") from last_exception
        raise ProviderError("Generation failed with unknown error")

    def count_tokens(self, text: str) -> int:
        """
        Estimate tokens in a text string.

        Ollama doesn't have a direct token counting endpoint.
        Uses rough estimation based on model type.

        Args:
            text: Text to count tokens for

        Returns:
            Token count (estimated)
        """
        # Most models use ~4 characters per token
        # Llama models tend to be slightly different
        if "llama" in self.model.lower():
            return len(text) // 4
        return len(text) // 4

    def list_models(self) -> list[str]:
        """
        List available models from Ollama.

        Returns:
            List of model names
        """
        self.ensure_initialized()

        try:
            models = self._client.list()
            return [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list models: {e}")
            return []

    def pull_model(self, model_name: str | None = None) -> bool:
        """
        Pull/download a model to Ollama.

        Args:
            model_name: Model to pull (defaults to self.model)

        Returns:
            True if successful
        """
        self.ensure_initialized()

        model = model_name or self.model
        try:
            logger.info(f"Pulling model: {model}")
            self._client.pull(model)
            logger.info(f"Successfully pulled model: {model}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {e}")
            return False
