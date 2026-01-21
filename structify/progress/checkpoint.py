"""Checkpoint system for resumable execution."""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from structify.core.exceptions import CheckpointError
from structify.utils.logging import get_logger

logger = get_logger("checkpoint")


@dataclass
class StageCheckpoint:
    """Checkpoint data for a single stage."""

    status: str = "pending"  # pending, in_progress, completed, error
    total_items: int = 0
    completed_items: int = 0
    last_completed_item: str = ""
    records_extracted: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error_message: str | None = None
    extra_data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """
    Complete checkpoint state for a pipeline run.

    Stores all information needed to resume from any point.
    """

    pipeline_id: str
    input_path: str
    started_at: str
    last_checkpoint: str
    stages: dict[str, StageCheckpoint] = field(default_factory=dict)
    schema_hash: str | None = None
    config_hash: str | None = None
    partial_results_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            "pipeline_id": self.pipeline_id,
            "input_path": self.input_path,
            "started_at": self.started_at,
            "last_checkpoint": self.last_checkpoint,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "schema_hash": self.schema_hash,
            "config_hash": self.config_hash,
            "partial_results_file": self.partial_results_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary."""
        stages = {}
        for name, stage_data in data.get("stages", {}).items():
            stages[name] = StageCheckpoint(**stage_data)

        return cls(
            pipeline_id=data["pipeline_id"],
            input_path=data["input_path"],
            started_at=data["started_at"],
            last_checkpoint=data["last_checkpoint"],
            stages=stages,
            schema_hash=data.get("schema_hash"),
            config_hash=data.get("config_hash"),
            partial_results_file=data.get("partial_results_file"),
        )


class CheckpointManager:
    """
    Manages checkpoints for resumable pipeline execution.

    Creates and manages:
    - Pipeline state files
    - Stage checkpoints
    - Partial results
    """

    def __init__(self, state_dir: str | Path = ".structify_state"):
        """
        Initialize the checkpoint manager.

        Args:
            state_dir: Directory to store checkpoint files
        """
        self.state_dir = Path(state_dir)
        self._checkpoint: Checkpoint | None = None
        self._auto_save = True

    def initialize(self, input_path: str | Path, pipeline_id: str | None = None) -> Checkpoint:
        """
        Initialize a new checkpoint or load existing one.

        Args:
            input_path: Path to input data
            pipeline_id: Optional pipeline ID (auto-generated if not provided)

        Returns:
            Checkpoint object
        """
        input_path = str(Path(input_path).resolve())

        # Check for existing checkpoint
        existing = self.find_checkpoint(input_path)
        if existing:
            logger.info(f"Found existing checkpoint: {existing.pipeline_id}")
            self._checkpoint = existing
            return existing

        # Create new checkpoint
        if pipeline_id is None:
            pipeline_id = self._generate_pipeline_id(input_path)

        now = datetime.now().isoformat()
        self._checkpoint = Checkpoint(
            pipeline_id=pipeline_id,
            input_path=input_path,
            started_at=now,
            last_checkpoint=now,
        )

        # Create state directory
        self._get_checkpoint_dir().mkdir(parents=True, exist_ok=True)

        logger.info(f"Created new checkpoint: {pipeline_id}")
        self.save()

        return self._checkpoint

    def _generate_pipeline_id(self, input_path: str) -> str:
        """Generate a unique pipeline ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_hash = hashlib.md5(input_path.encode()).hexdigest()[:8]
        return f"{timestamp}_{path_hash}"

    def _get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory for the current pipeline."""
        if self._checkpoint is None:
            raise CheckpointError("No checkpoint initialized")
        return self.state_dir / self._checkpoint.pipeline_id

    def _get_state_file(self) -> Path:
        """Get the main state file path."""
        return self._get_checkpoint_dir() / "pipeline_state.json"

    def find_checkpoint(self, input_path: str | Path) -> Checkpoint | None:
        """
        Find an existing checkpoint for the given input path.

        Args:
            input_path: Path to input data

        Returns:
            Checkpoint if found, None otherwise
        """
        input_path = str(Path(input_path).resolve())

        if not self.state_dir.exists():
            return None

        # Look through all checkpoint directories
        for checkpoint_dir in self.state_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue

            state_file = checkpoint_dir / "pipeline_state.json"
            if not state_file.exists():
                continue

            try:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if data.get("input_path") == input_path:
                    return Checkpoint.from_dict(data)
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def save(self) -> None:
        """Save the current checkpoint to disk."""
        if self._checkpoint is None:
            raise CheckpointError("No checkpoint to save")

        self._checkpoint.last_checkpoint = datetime.now().isoformat()

        state_file = self._get_state_file()
        state_file.parent.mkdir(parents=True, exist_ok=True)

        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(self._checkpoint.to_dict(), f, indent=2)

        logger.debug(f"Checkpoint saved: {state_file}")

    def load(self, checkpoint_dir: str | Path) -> Checkpoint:
        """
        Load a checkpoint from a directory.

        Args:
            checkpoint_dir: Directory containing checkpoint files

        Returns:
            Checkpoint object
        """
        checkpoint_dir = Path(checkpoint_dir)
        state_file = checkpoint_dir / "pipeline_state.json"

        if not state_file.exists():
            raise CheckpointError(f"Checkpoint file not found: {state_file}")

        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._checkpoint = Checkpoint.from_dict(data)
        return self._checkpoint

    def update_stage(
        self,
        stage_name: str,
        status: str | None = None,
        completed_items: int | None = None,
        last_completed_item: str | None = None,
        records_extracted: int | None = None,
        error_message: str | None = None,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """
        Update checkpoint for a stage.

        Args:
            stage_name: Name of the stage
            status: Stage status
            completed_items: Number of completed items
            last_completed_item: Last completed item identifier
            records_extracted: Total records extracted
            error_message: Error message if status is error
            extra_data: Additional data to store
        """
        if self._checkpoint is None:
            raise CheckpointError("No checkpoint initialized")

        if stage_name not in self._checkpoint.stages:
            self._checkpoint.stages[stage_name] = StageCheckpoint()

        stage = self._checkpoint.stages[stage_name]

        if status is not None:
            stage.status = status
            if status == "in_progress" and stage.started_at is None:
                stage.started_at = datetime.now().isoformat()
            elif status == "completed":
                stage.completed_at = datetime.now().isoformat()

        if completed_items is not None:
            stage.completed_items = completed_items

        if last_completed_item is not None:
            stage.last_completed_item = last_completed_item

        if records_extracted is not None:
            stage.records_extracted = records_extracted

        if error_message is not None:
            stage.error_message = error_message

        if extra_data is not None:
            stage.extra_data.update(extra_data)

        if self._auto_save:
            self.save()

    def start_stage(self, stage_name: str, total_items: int) -> StageCheckpoint:
        """
        Start a new stage.

        Args:
            stage_name: Name of the stage
            total_items: Total items to process

        Returns:
            StageCheckpoint object
        """
        if self._checkpoint is None:
            raise CheckpointError("No checkpoint initialized")

        if stage_name not in self._checkpoint.stages:
            self._checkpoint.stages[stage_name] = StageCheckpoint()

        stage = self._checkpoint.stages[stage_name]
        stage.total_items = total_items
        stage.status = "in_progress"
        stage.started_at = datetime.now().isoformat()

        if self._auto_save:
            self.save()

        return stage

    def complete_stage(self, stage_name: str, records_extracted: int = 0) -> None:
        """
        Mark a stage as completed.

        Args:
            stage_name: Name of the stage
            records_extracted: Total records extracted
        """
        self.update_stage(
            stage_name,
            status="completed",
            records_extracted=records_extracted,
        )

    def get_stage(self, stage_name: str) -> StageCheckpoint | None:
        """
        Get checkpoint for a stage.

        Args:
            stage_name: Name of the stage

        Returns:
            StageCheckpoint or None
        """
        if self._checkpoint is None:
            return None
        return self._checkpoint.stages.get(stage_name)

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage is completed."""
        stage = self.get_stage(stage_name)
        return stage is not None and stage.status == "completed"

    def get_pending_items(self, stage_name: str, all_items: list[str]) -> list[str]:
        """
        Get items that haven't been processed yet.

        Args:
            stage_name: Name of the stage
            all_items: List of all items

        Returns:
            List of pending items
        """
        stage = self.get_stage(stage_name)
        if stage is None or not stage.last_completed_item:
            return all_items

        # Find the last completed item and return everything after it
        try:
            last_idx = all_items.index(stage.last_completed_item)
            return all_items[last_idx + 1:]
        except ValueError:
            return all_items

    def set_schema_hash(self, schema_hash: str) -> None:
        """Store the schema hash for validation on resume."""
        if self._checkpoint:
            self._checkpoint.schema_hash = schema_hash
            self.save()

    def clear(self) -> None:
        """Clear the current checkpoint."""
        if self._checkpoint is None:
            return

        checkpoint_dir = self._get_checkpoint_dir()
        if checkpoint_dir.exists():
            import shutil
            shutil.rmtree(checkpoint_dir)

        self._checkpoint = None
        logger.info("Checkpoint cleared")

    @property
    def checkpoint(self) -> Checkpoint | None:
        """Get the current checkpoint."""
        return self._checkpoint

    @property
    def auto_save(self) -> bool:
        """Get auto-save setting."""
        return self._auto_save

    @auto_save.setter
    def auto_save(self, value: bool) -> None:
        """Set auto-save setting."""
        self._auto_save = value
