"""Base class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLMProvider(ABC):
    """
    Base class for LLM providers.

    Providers handle communication with specific LLM APIs (Gemini, OpenAI, etc.).
    """

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
        Initialize the provider.

        Args:
            api_key: API key for authentication
            model: Model name to use
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            between_calls_delay: Delay between normal API calls
            temperature: Generation temperature
            max_output_tokens: Maximum output tokens
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.between_calls_delay = between_calls_delay
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider with API credentials."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        file_ref: Any | None = None,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send
            file_ref: Optional file reference (from upload_file)

        Returns:
            The LLM's response text
        """
        pass

    @abstractmethod
    def upload_file(self, file_path: str) -> Any:
        """
        Upload a file to the provider.

        Args:
            file_path: Path to the file

        Returns:
            File reference object
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._is_initialized

    def ensure_initialized(self) -> None:
        """Ensure provider is initialized, initialize if not."""
        if not self._is_initialized:
            self.initialize()
