"""LLM providers for structify.

Providers handle communication with LLM APIs for content generation.
Each provider implements the BaseLLMProvider interface.

Available providers:
- GeminiProvider: Google Gemini API (default)
- AnthropicProvider: Anthropic Claude API
- OpenRouterProvider: OpenRouter API (access to 100+ models)
- OllamaProvider: Local LLM inference via Ollama
"""

from structify.providers.base import BaseLLMProvider
from structify.providers.gemini import GeminiProvider

# Lazy imports for optional providers to avoid import errors
# when their dependencies aren't installed


def _get_anthropic_provider():
    from structify.providers.anthropic import AnthropicProvider
    return AnthropicProvider


def _get_openrouter_provider():
    from structify.providers.openrouter import OpenRouterProvider
    return OpenRouterProvider


def _get_ollama_provider():
    from structify.providers.ollama import OllamaProvider
    return OllamaProvider


# Export classes (with lazy loading for optional deps)
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


def get_provider(provider_name: str, **kwargs) -> BaseLLMProvider:
    """
    Factory function to get a provider by name.

    Args:
        provider_name: Name of the provider (gemini, anthropic, openrouter, ollama)
        **kwargs: Provider-specific configuration

    Returns:
        Initialized provider instance

    Example:
        >>> provider = get_provider("anthropic", model="claude-sonnet-4-20250514")
        >>> provider = get_provider("ollama", model="llama3.2")
        >>> provider = get_provider("openrouter", model="anthropic/claude-sonnet-4")
    """
    providers = {
        "gemini": GeminiProvider,
        "google": GeminiProvider,
        "anthropic": _get_anthropic_provider,
        "claude": _get_anthropic_provider,
        "openrouter": _get_openrouter_provider,
        "ollama": _get_ollama_provider,
        "local": _get_ollama_provider,
    }

    provider_name = provider_name.lower().strip()

    if provider_name not in providers:
        available = list(providers.keys())
        raise ValueError(
            f"Unknown provider: {provider_name}. Available: {available}"
        )

    provider_class = providers[provider_name]

    # Handle lazy loading
    if callable(provider_class) and not isinstance(provider_class, type):
        try:
            provider_class = provider_class()
        except ImportError as e:
            raise ImportError(
                f"Provider '{provider_name}' requires additional dependencies. "
                f"Install with: pip install pdf-structify[{provider_name}]"
            ) from e

    return provider_class(**kwargs)


__all__ = [
    "BaseLLMProvider",
    "GeminiProvider",
    "AnthropicProvider",
    "OpenRouterProvider",
    "OllamaProvider",
    "get_provider",
]
