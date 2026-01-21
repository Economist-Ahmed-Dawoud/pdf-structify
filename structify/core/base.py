"""Base classes for structify components."""

from abc import ABC, abstractmethod
from typing import Any, TypeVar, Generic

T = TypeVar("T")
R = TypeVar("R")


class BaseTransformer(ABC, Generic[T, R]):
    """
    Base class for all transformers following sklearn-like API.

    Transformers implement fit(), transform(), and fit_transform() methods.
    """

    def __init__(self, **params):
        """Initialize transformer with parameters."""
        self._params = params
        self._is_fitted = False

    def get_params(self) -> dict[str, Any]:
        """Get transformer parameters."""
        return self._params.copy()

    def set_params(self, **params) -> "BaseTransformer":
        """Set transformer parameters."""
        self._params.update(params)
        return self

    @abstractmethod
    def fit(self, data: T, **kwargs) -> "BaseTransformer":
        """
        Fit the transformer to the data.

        Args:
            data: Input data to fit on
            **kwargs: Additional arguments

        Returns:
            self
        """
        pass

    @abstractmethod
    def transform(self, data: T, **kwargs) -> R:
        """
        Transform the data.

        Args:
            data: Input data to transform
            **kwargs: Additional arguments

        Returns:
            Transformed data
        """
        pass

    def fit_transform(self, data: T, **kwargs) -> R:
        """
        Fit and transform in one step.

        Args:
            data: Input data
            **kwargs: Additional arguments

        Returns:
            Transformed data
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    @property
    def is_fitted(self) -> bool:
        """Check if transformer has been fitted."""
        return self._is_fitted


class BaseExtractor(BaseTransformer[T, R]):
    """
    Base class for data extractors.

    Extractors are transformers that extract structured data from documents.
    """

    def __init__(self, **params):
        super().__init__(**params)
        self._schema = None

    @property
    def schema(self):
        """Get the extraction schema."""
        return self._schema

    @schema.setter
    def schema(self, value):
        """Set the extraction schema."""
        self._schema = value


class BaseLLMProvider(ABC):
    """
    Base class for LLM providers.

    Providers handle communication with specific LLM APIs.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_retries: int = 5,
        retry_delay: int = 60,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the provider with API credentials."""
        pass

    @abstractmethod
    def generate(self, prompt: str, file_path: str | None = None) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The prompt to send
            file_path: Optional path to a file (e.g., PDF) to include

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
