"""Structured logging for structify."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme


# Custom theme for structify
STRUCTIFY_THEME = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "stage": "bold magenta",
    "progress": "blue",
})

# Global console instance
console = Console(theme=STRUCTIFY_THEME)


class StructifyFormatter(logging.Formatter):
    """Custom formatter for structify logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record."""
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname.ljust(7)
        module = record.name.split(".")[-1].ljust(15)

        return f"{timestamp} | {level} | {module} | {record.getMessage()}"


class Logger:
    """
    Centralized logging configuration for structify.

    Supports both console (with rich formatting) and file logging.
    """

    _instance: "Logger | None" = None
    _initialized: bool = False

    def __init__(self):
        self.console_level = logging.INFO
        self.file_level = logging.DEBUG
        self.log_dir = Path("logs")
        self.log_format = "detailed"
        self._handlers: list[logging.Handler] = []
        self._loggers: dict[str, logging.Logger] = {}

    @classmethod
    def get_instance(cls) -> "Logger":
        """Get the singleton logger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def setup(
        cls,
        console_level: str = "INFO",
        file_level: str = "DEBUG",
        log_dir: str = "logs",
        log_format: str = "detailed",
    ) -> "Logger":
        """
        Configure the logging system.

        Args:
            console_level: Logging level for console output
            file_level: Logging level for file output
            log_dir: Directory for log files
            log_format: Format style ("simple", "detailed", or "json")

        Returns:
            The logger instance
        """
        instance = cls.get_instance()

        instance.console_level = getattr(logging, console_level.upper())
        instance.file_level = getattr(logging, file_level.upper())
        instance.log_dir = Path(log_dir)
        instance.log_format = log_format

        # Create log directory
        instance.log_dir.mkdir(parents=True, exist_ok=True)

        # Configure root logger
        root_logger = logging.getLogger("structify")
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in instance._handlers:
            root_logger.removeHandler(handler)
        instance._handlers.clear()

        # Console handler with rich formatting
        console_handler = RichHandler(
            console=console,
            show_time=False,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
        )
        console_handler.setLevel(instance.console_level)
        root_logger.addHandler(console_handler)
        instance._handlers.append(console_handler)

        # File handler
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_file = instance.log_dir / f"structify_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(instance.file_level)
        file_handler.setFormatter(StructifyFormatter())
        root_logger.addHandler(file_handler)
        instance._handlers.append(file_handler)

        cls._initialized = True
        return instance

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger with the given name.

        Args:
            name: Logger name (will be prefixed with 'structify.')

        Returns:
            Logger instance
        """
        instance = cls.get_instance()

        if not cls._initialized:
            cls.setup()

        full_name = f"structify.{name}" if not name.startswith("structify.") else name

        if full_name not in instance._loggers:
            instance._loggers[full_name] = logging.getLogger(full_name)

        return instance._loggers[full_name]

    @classmethod
    def log_stage(cls, stage_name: str, stage_num: int, total_stages: int) -> None:
        """
        Log the start of a pipeline stage.

        Args:
            stage_name: Name of the stage
            stage_num: Current stage number
            total_stages: Total number of stages
        """
        console.print()
        console.print(f"[stage]{'─' * 60}[/stage]")
        console.print(f"[stage]Stage {stage_num}/{total_stages}: {stage_name}[/stage]")
        console.print(f"[stage]{'─' * 60}[/stage]")

    @classmethod
    def log_success(cls, message: str) -> None:
        """Log a success message."""
        console.print(f"[success]✓ {message}[/success]")

    @classmethod
    def log_error(cls, message: str) -> None:
        """Log an error message."""
        console.print(f"[error]✗ {message}[/error]")

    @classmethod
    def log_warning(cls, message: str) -> None:
        """Log a warning message."""
        console.print(f"[warning]⚠ {message}[/warning]")

    @classmethod
    def log_info(cls, message: str) -> None:
        """Log an info message."""
        console.print(f"[info]{message}[/info]")


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return Logger.get_logger(name)
