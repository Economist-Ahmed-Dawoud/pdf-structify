"""Configuration management for structify."""

import os
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from dotenv import load_dotenv
import yaml


@dataclass
class Config:
    """
    Global configuration for structify.

    Supports loading from environment variables, .env files, or YAML config files.
    """

    # API Keys
    gemini_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None

    # Ollama settings (no API key required)
    ollama_base_url: str = "http://localhost:11434"

    # Default provider settings
    # Supported providers: gemini, anthropic, openrouter, ollama
    default_provider: str = "gemini"
    default_model: str = "gemini-2.0-flash"

    # Request settings
    timeout: int = 120
    max_retries: int = 5
    retry_delay: int = 60
    between_calls_delay: int = 3

    # Generation settings
    temperature: float = 0.1
    max_output_tokens: int = 60000

    # PDF splitting settings
    pages_per_chunk: int = 10

    # Schema detection settings
    sample_ratio: float = 0.1
    max_samples: int = 30

    # Logging settings
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_format: str = "detailed"

    # State/checkpoint settings
    state_dir: str = ".structify_state"
    enable_checkpoints: bool = True

    # Cache settings
    cache_dir: str = ".structify_cache"
    enable_cache: bool = True

    # Instance storage
    _instance: "Config | None" = field(default=None, repr=False, compare=False)

    @classmethod
    def get_instance(cls) -> "Config":
        """Get the singleton config instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def set(cls, **kwargs) -> "Config":
        """
        Set configuration values.

        Args:
            **kwargs: Configuration key-value pairs

        Returns:
            The config instance
        """
        instance = cls.get_instance()
        for key, value in kwargs.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                raise ValueError(f"Unknown configuration key: {key}")
        return instance

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value
        """
        instance = cls.get_instance()
        return getattr(instance, key, default)

    @classmethod
    def from_env(cls, dotenv_path: str | Path | None = None) -> "Config":
        """
        Load configuration from environment variables.

        Args:
            dotenv_path: Optional path to .env file

        Returns:
            The config instance
        """
        if dotenv_path:
            load_dotenv(dotenv_path)
        else:
            load_dotenv()

        instance = cls.get_instance()

        # Map environment variables to config
        env_mapping = {
            "GEMINI_API_KEY": "gemini_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "ANTHROPIC_API_KEY": "anthropic_api_key",
            "OPENROUTER_API_KEY": "openrouter_api_key",
            "OLLAMA_BASE_URL": "ollama_base_url",
            "STRUCTIFY_PROVIDER": "default_provider",
            "STRUCTIFY_MODEL": "default_model",
            "STRUCTIFY_TIMEOUT": ("timeout", int),
            "STRUCTIFY_MAX_RETRIES": ("max_retries", int),
            "STRUCTIFY_RETRY_DELAY": ("retry_delay", int),
            "STRUCTIFY_TEMPERATURE": ("temperature", float),
            "STRUCTIFY_PAGES_PER_CHUNK": ("pages_per_chunk", int),
            "STRUCTIFY_LOG_LEVEL": "log_level",
            "STRUCTIFY_LOG_DIR": "log_dir",
            "STRUCTIFY_STATE_DIR": "state_dir",
        }

        for env_key, config_key in env_mapping.items():
            value = os.getenv(env_key)
            if value is not None:
                if isinstance(config_key, tuple):
                    attr_name, converter = config_key
                    setattr(instance, attr_name, converter(value))
                else:
                    setattr(instance, config_key, value)

        cls._instance = instance
        return instance

    @classmethod
    def from_file(cls, config_path: str | Path) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML config file

        Returns:
            The config instance
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        instance = cls.get_instance()
        for key, value in config_data.items():
            if hasattr(instance, key):
                setattr(instance, key, value)

        cls._instance = instance
        return instance

    @classmethod
    def reset(cls) -> None:
        """Reset configuration to defaults."""
        cls._instance = None

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def save(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save the config file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Don't save sensitive API keys
        config_dict = self.to_dict()
        for key in ["gemini_api_key", "openai_api_key", "anthropic_api_key", "openrouter_api_key"]:
            if key in config_dict:
                config_dict[key] = None

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
