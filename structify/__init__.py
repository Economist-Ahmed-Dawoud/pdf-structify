"""
Structify - Extract structured data from PDFs using LLMs

A scikit-learn-style library for extracting structured data from PDF documents
using Large Language Models (LLMs).

Quick Start:
    >>> from structify import Pipeline
    >>> pipeline = Pipeline.quick_start()
    >>> results = pipeline.fit_transform("my_pdfs/")
    >>> results.to_csv("output.csv")

With custom schema:
    >>> from structify import Pipeline, SchemaBuilder
    >>> schema = SchemaBuilder.from_description('''
    ...     Extract research findings:
    ...     - Author names and year
    ...     - Main numerical finding
    ...     - Statistical significance
    ... ''')
    >>> pipeline = Pipeline.from_schema(schema)
    >>> results = pipeline.fit_transform("papers/")
"""

__version__ = "0.1.18"
__author__ = "Ahmed Dawood"

# Core configuration
from structify.core.config import Config
from structify.core.exceptions import (
    StructifyError,
    ConfigurationError,
    ExtractionError,
    SchemaError,
    ProviderError,
    CheckpointError,
    InvalidResponseError,
    RateLimitError,
)

# Preprocessing
from structify.preprocessing.splitter import PDFSplitter
from structify.preprocessing.loader import PDFLoader, PDFDocument, PDFChunk

# Schema
from structify.schema.types import Schema, Field, FieldType
from structify.schema.builder import SchemaBuilder
from structify.schema.detector import SchemaDetector, SchemaReviewer, ExtractionPurpose

# Providers
from structify.providers.gemini import GeminiProvider
from structify.providers import get_provider

# Optional providers (lazy loaded to avoid import errors)
try:
    from structify.providers.anthropic import AnthropicProvider
except ImportError:
    AnthropicProvider = None  # type: ignore

try:
    from structify.providers.openrouter import OpenRouterProvider
except ImportError:
    OpenRouterProvider = None  # type: ignore

try:
    from structify.providers.ollama import OllamaProvider
except ImportError:
    OllamaProvider = None  # type: ignore

# Extractors
from structify.extractors.extractor import LLMExtractor, MultiExtractor
from structify.extractors.prompt_generator import PromptGenerator

# Pipeline
from structify.pipeline.pipeline import Pipeline

# Progress & State
from structify.progress.tracker import ProgressTracker
from structify.progress.checkpoint import CheckpointManager
from structify.state.manager import StateManager

# Output
from structify.output.writers import OutputWriter

# Logging
from structify.utils.logging import Logger, get_logger

__all__ = [
    # Version
    "__version__",
    # Configuration
    "Config",
    # Exceptions
    "StructifyError",
    "ConfigurationError",
    "ExtractionError",
    "SchemaError",
    "ProviderError",
    "CheckpointError",
    "InvalidResponseError",
    "RateLimitError",
    # Preprocessing
    "PDFSplitter",
    "PDFLoader",
    "PDFDocument",
    "PDFChunk",
    # Schema
    "Schema",
    "Field",
    "FieldType",
    "SchemaBuilder",
    "SchemaDetector",
    "SchemaReviewer",
    "ExtractionPurpose",
    # Providers
    "GeminiProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "get_provider",
    # Extractors
    "LLMExtractor",
    "MultiExtractor",
    "PromptGenerator",
    # Pipeline
    "Pipeline",
    # Progress & State
    "ProgressTracker",
    "CheckpointManager",
    "StateManager",
    # Output
    "OutputWriter",
    # Logging
    "Logger",
    "get_logger",
]


def configure(**kwargs) -> Config:
    """
    Configure structify settings.

    Args:
        **kwargs: Configuration key-value pairs

    Returns:
        Config instance

    Example:
        >>> import structify
        >>> # Configure with Gemini (default)
        >>> structify.configure(
        ...     gemini_api_key="your-api-key",
        ...     pages_per_chunk=10,
        ... )
        >>> # Configure with Anthropic Claude
        >>> structify.configure(
        ...     default_provider="anthropic",
        ...     anthropic_api_key="your-api-key",
        ...     default_model="claude-sonnet-4-20250514",
        ... )
        >>> # Configure with OpenRouter
        >>> structify.configure(
        ...     default_provider="openrouter",
        ...     openrouter_api_key="your-api-key",
        ...     default_model="anthropic/claude-sonnet-4",
        ... )
        >>> # Configure with Ollama (local)
        >>> structify.configure(
        ...     default_provider="ollama",
        ...     ollama_base_url="http://localhost:11434",
        ...     default_model="llama3.2",
        ... )
    """
    return Config.set(**kwargs)


def from_env(dotenv_path: str | None = None) -> Config:
    """
    Load configuration from environment variables.

    Args:
        dotenv_path: Optional path to .env file

    Returns:
        Config instance
    """
    return Config.from_env(dotenv_path)
