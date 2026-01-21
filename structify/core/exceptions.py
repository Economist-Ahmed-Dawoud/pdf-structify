"""Custom exceptions for the structify package."""


class StructifyError(Exception):
    """Base exception for all structify errors."""
    pass


class ConfigurationError(StructifyError):
    """Raised when there's a configuration error."""
    pass


class ExtractionError(StructifyError):
    """Raised when extraction fails."""
    pass


class SchemaError(StructifyError):
    """Raised when there's a schema-related error."""
    pass


class ProviderError(StructifyError):
    """Raised when there's an LLM provider error."""
    pass


class CheckpointError(StructifyError):
    """Raised when checkpoint operations fail."""
    pass


class InvalidResponseError(ExtractionError):
    """Raised when LLM response is invalid or cannot be parsed."""
    pass


class RateLimitError(ProviderError):
    """Raised when API rate limit is hit."""
    pass


class TimeoutError(ProviderError):
    """Raised when API request times out (504 Deadline Exceeded)."""
    pass


class ValidationError(StructifyError):
    """Raised when data validation fails."""
    pass
