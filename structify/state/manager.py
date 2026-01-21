"""State manager for pipeline execution."""

import signal
import atexit
from pathlib import Path
from typing import Any, Callable

from structify.progress.checkpoint import CheckpointManager, Checkpoint
from structify.utils.logging import get_logger, Logger

logger = get_logger("state")


class StateManager:
    """
    Manages execution state and handles interrupts gracefully.

    Features:
    - Automatic checkpoint saving on interrupt (Ctrl+C)
    - Graceful shutdown handling
    - State persistence across sessions
    """

    def __init__(
        self,
        state_dir: str | Path = ".structify_state",
        enable_checkpoints: bool = True,
    ):
        """
        Initialize the state manager.

        Args:
            state_dir: Directory for state files
            enable_checkpoints: Whether to enable checkpoint saving
        """
        self.state_dir = Path(state_dir)
        self.enable_checkpoints = enable_checkpoints
        self.checkpoint_manager = CheckpointManager(state_dir)

        self._original_sigint = None
        self._original_sigterm = None
        self._shutdown_callbacks: list[Callable[[], None]] = []
        self._is_running = False
        self._interrupted = False

    def start(self, input_path: str | Path, pipeline_id: str | None = None) -> Checkpoint:
        """
        Start state management for a pipeline run.

        Args:
            input_path: Path to input data
            pipeline_id: Optional pipeline ID

        Returns:
            Checkpoint object
        """
        self._is_running = True
        self._interrupted = False

        # Register signal handlers
        self._register_signal_handlers()

        # Register cleanup on exit
        atexit.register(self._cleanup)

        # Initialize checkpoint
        if self.enable_checkpoints:
            checkpoint = self.checkpoint_manager.initialize(input_path, pipeline_id)
            logger.info(f"State management started: {checkpoint.pipeline_id}")
            return checkpoint

        return Checkpoint(
            pipeline_id="no-checkpoint",
            input_path=str(input_path),
            started_at="",
            last_checkpoint="",
        )

    def stop(self) -> None:
        """Stop state management and save final state."""
        if not self._is_running:
            return

        self._is_running = False

        # Restore original signal handlers
        self._restore_signal_handlers()

        # Save final checkpoint
        if self.enable_checkpoints:
            self.checkpoint_manager.save()

        # Remove atexit handler
        atexit.unregister(self._cleanup)

        logger.info("State management stopped")

    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        try:
            self._original_sigint = signal.signal(signal.SIGINT, self._handle_interrupt)
            self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_interrupt)
        except (ValueError, OSError):
            # Signal handling not available (e.g., in threads)
            pass

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        try:
            if self._original_sigint is not None:
                signal.signal(signal.SIGINT, self._original_sigint)
            if self._original_sigterm is not None:
                signal.signal(signal.SIGTERM, self._original_sigterm)
        except (ValueError, OSError):
            pass

    def _handle_interrupt(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal (Ctrl+C)."""
        if self._interrupted:
            # Second interrupt - force exit
            logger.warning("Force exit requested")
            raise KeyboardInterrupt()

        self._interrupted = True
        Logger.log_warning("Interrupt received. Saving checkpoint...")

        # Save checkpoint
        if self.enable_checkpoints:
            try:
                self.checkpoint_manager.save()
                Logger.log_success("Checkpoint saved. Run again to resume.")
            except Exception as e:
                Logger.log_error(f"Failed to save checkpoint: {e}")

        # Run shutdown callbacks
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception:
                pass

        raise KeyboardInterrupt()

    def _cleanup(self) -> None:
        """Cleanup on exit."""
        if self._is_running and self.enable_checkpoints:
            try:
                self.checkpoint_manager.save()
            except Exception:
                pass

    def add_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """
        Add a callback to run on shutdown.

        Args:
            callback: Function to call on shutdown
        """
        self._shutdown_callbacks.append(callback)

    def remove_shutdown_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove a shutdown callback.

        Args:
            callback: Function to remove
        """
        if callback in self._shutdown_callbacks:
            self._shutdown_callbacks.remove(callback)

    def update_stage(self, stage_name: str, **kwargs) -> None:
        """
        Update stage checkpoint.

        Args:
            stage_name: Name of the stage
            **kwargs: Stage checkpoint fields to update
        """
        if self.enable_checkpoints:
            self.checkpoint_manager.update_stage(stage_name, **kwargs)

    def start_stage(self, stage_name: str, total_items: int) -> None:
        """
        Start a new stage.

        Args:
            stage_name: Name of the stage
            total_items: Total items to process
        """
        if self.enable_checkpoints:
            self.checkpoint_manager.start_stage(stage_name, total_items)

    def complete_stage(self, stage_name: str, records_extracted: int = 0) -> None:
        """
        Mark a stage as completed.

        Args:
            stage_name: Name of the stage
            records_extracted: Total records extracted
        """
        if self.enable_checkpoints:
            self.checkpoint_manager.complete_stage(stage_name, records_extracted)

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed."""
        if not self.enable_checkpoints:
            return False
        return self.checkpoint_manager.is_stage_completed(stage_name)

    def get_pending_items(self, stage_name: str, all_items: list[str]) -> list[str]:
        """Get items that haven't been processed yet."""
        if not self.enable_checkpoints:
            return all_items
        return self.checkpoint_manager.get_pending_items(stage_name, all_items)

    def save_checkpoint(self) -> None:
        """Force save the current checkpoint."""
        if self.enable_checkpoints:
            self.checkpoint_manager.save()

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints for the current pipeline."""
        if self.enable_checkpoints:
            self.checkpoint_manager.clear()

    @property
    def checkpoint(self) -> Checkpoint | None:
        """Get the current checkpoint."""
        return self.checkpoint_manager.checkpoint

    @property
    def is_running(self) -> bool:
        """Check if state management is active."""
        return self._is_running

    @property
    def was_interrupted(self) -> bool:
        """Check if the pipeline was interrupted."""
        return self._interrupted

    def __enter__(self) -> "StateManager":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.stop()
        return False
