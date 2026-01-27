"""Main pipeline class for structify."""

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from structify.core.config import Config
from structify.preprocessing.splitter import PDFSplitter
from structify.preprocessing.loader import PDFLoader
from structify.schema.types import Schema
from structify.schema.builder import SchemaBuilder
from structify.schema.detector import SchemaDetector, SchemaReviewer, DetectionMode, ExtractionPurpose
from structify.extractors.extractor import LLMExtractor
from structify.providers import BaseLLMProvider, get_provider
from structify.progress.tracker import ProgressTracker
from structify.progress.checkpoint import CheckpointManager
from structify.state.manager import StateManager
from structify.output.writers import OutputWriter
from structify.utils.logging import get_logger, Logger

logger = get_logger("pipeline")


class Pipeline:
    """
    Sklearn-style pipeline for PDF data extraction.

    Chains together preprocessing, schema detection, and extraction
    steps into a single, easy-to-use interface.

    Example:
        >>> pipeline = Pipeline.quick_start()
        >>> results = pipeline.fit_transform("my_pdfs/")
        >>> results.to_csv("output.csv")
    """

    def __init__(
        self,
        steps: list[tuple[str, Any]] | None = None,
        schema: Schema | str | Path | None = None,
        provider: BaseLLMProvider | str | None = None,
        pages_per_chunk: int = 10,
        auto_split: bool = True,
        auto_detect_schema: bool = True,
        deduplicate: bool = True,
        enable_checkpoints: bool = True,
        state_dir: str = ".structify_state",
        detection_mode: DetectionMode = "moderate",
        purpose: ExtractionPurpose = "findings",
        seed: int | None = None,
        detection_model: str | None = None,
        extraction_model: str | None = None,
        extraction_sample_ratio: float | None = None,
        extraction_max_samples: int | None = None,
    ):
        """
        Initialize the pipeline.

        Args:
            steps: List of (name, transformer) tuples
            schema: Pre-defined schema - can be Schema object or path to JSON/YAML file.
                If provided, skips schema detection (resume capability).
            provider: LLM provider - can be a provider instance or provider name string:
                - "gemini" (default), "anthropic"/"claude", "openrouter", "ollama"/"local"
            pages_per_chunk: Pages per PDF chunk
            auto_split: Automatically split large PDFs
            auto_detect_schema: Automatically detect schema if not provided
            deduplicate: Deduplicate extraction results
            enable_checkpoints: Enable checkpoint/resume functionality
            state_dir: Directory for state files
            detection_mode: Schema detection mode - "strict" (UP TO 5-7 fields),
                "moderate" (UP TO 7-12 fields), or "extended" (UP TO 12-20 fields)
            purpose: What type of information to extract:
                - "findings": Research findings, estimates, coefficients (default)
                - "policies": Policies, incentives, decisions, interventions
            seed: Random seed for reproducible sampling (both detection and extraction)
            detection_model: Model to use for schema detection (e.g., "gemini-2.0-flash")
            extraction_model: Model to use for data extraction (e.g., "gemini-2.5-pro")
            extraction_sample_ratio: Fraction of documents to extract from (0.0-1.0)
            extraction_max_samples: Maximum number of documents to extract from
        """
        self.steps = steps or []
        self.provider = provider
        self.pages_per_chunk = pages_per_chunk
        self.auto_split = auto_split
        self.deduplicate = deduplicate
        self.enable_checkpoints = enable_checkpoints
        self.state_dir = state_dir
        self.detection_mode = detection_mode
        self.purpose = purpose
        self.seed = seed
        self.detection_model = detection_model
        self.extraction_model = extraction_model
        self.extraction_sample_ratio = extraction_sample_ratio
        self.extraction_max_samples = extraction_max_samples

        # Handle schema - can be object or path to file
        if isinstance(schema, (str, Path)):
            self._schema = Schema.load(schema)
            logger.info(f"Loaded schema from {schema}")
            self.auto_detect_schema = False  # Don't detect if schema provided
        else:
            self._schema = schema
            self.auto_detect_schema = auto_detect_schema if schema is None else False

        self._tracker: ProgressTracker | None = None
        self._state_manager: StateManager | None = None
        self._is_fitted = self._schema is not None  # Pre-fitted if schema loaded
        self._results: pd.DataFrame | None = None

    @classmethod
    def quick_start(
        cls,
        schema: Schema | str | Path | None = None,
        pages_per_chunk: int = 10,
        detection_mode: DetectionMode = "moderate",
        purpose: ExtractionPurpose = "findings",
        seed: int | None = None,
        detection_model: str | None = None,
        extraction_model: str | None = None,
        extraction_sample_ratio: float | None = None,
        extraction_max_samples: int | None = None,
    ) -> "Pipeline":
        """
        Create a pipeline with sensible defaults.

        Args:
            schema: Optional schema - can be Schema object or path to JSON/YAML
            pages_per_chunk: Pages per PDF chunk
            detection_mode: Schema detection mode - "strict", "moderate", or
                "extended" (these are maximum limits, not targets)
            purpose: What to extract - "findings" (research results) or
                "policies" (interventions, incentives, decisions)
            seed: Random seed for reproducible sampling
            detection_model: Model for schema detection (e.g., "gemini-2.0-flash")
            extraction_model: Model for data extraction (e.g., "gemini-2.5-pro")
            extraction_sample_ratio: Fraction of documents to extract from
            extraction_max_samples: Maximum documents to extract from

        Returns:
            Configured pipeline
        """
        return cls(
            schema=schema,
            pages_per_chunk=pages_per_chunk,
            auto_split=True,
            deduplicate=True,
            enable_checkpoints=True,
            detection_mode=detection_mode,
            purpose=purpose,
            seed=seed,
            detection_model=detection_model,
            extraction_model=extraction_model,
            extraction_sample_ratio=extraction_sample_ratio,
            extraction_max_samples=extraction_max_samples,
        )

    @classmethod
    def from_schema(cls, schema: Schema | str | Path, **kwargs) -> "Pipeline":
        """
        Create a pipeline with a specific schema.

        Args:
            schema: Schema object or path to schema file
            **kwargs: Additional pipeline arguments

        Returns:
            Configured pipeline
        """
        if isinstance(schema, (str, Path)):
            schema = Schema.load(schema)

        return cls(schema=schema, auto_detect_schema=False, **kwargs)

    @classmethod
    def from_description(cls, description: str, provider: str | None = None, **kwargs) -> "Pipeline":
        """
        Create a pipeline from a natural language description.

        Args:
            description: Natural language description of what to extract
            provider: Provider name (gemini, anthropic, openrouter, ollama)
            **kwargs: Additional pipeline arguments

        Returns:
            Configured pipeline
        """
        provider_name = provider or Config.get("default_provider") or "gemini"
        provider_instance = get_provider(provider_name)
        builder = SchemaBuilder(provider=provider_instance)
        schema = builder.from_description(description)

        return cls(schema=schema, auto_detect_schema=False, provider=provider_instance, **kwargs)

    @classmethod
    def resume(cls, input_path: str | Path, state_dir: str = ".structify_state") -> "Pipeline":
        """
        Resume a pipeline from a checkpoint.

        Args:
            input_path: Original input path
            state_dir: State directory

        Returns:
            Pipeline configured to resume
        """
        checkpoint_manager = CheckpointManager(state_dir)
        checkpoint = checkpoint_manager.find_checkpoint(str(input_path))

        if checkpoint is None:
            logger.warning("No checkpoint found, starting fresh")
            return cls.quick_start()

        logger.info(f"Resuming pipeline: {checkpoint.pipeline_id}")

        pipeline = cls(
            enable_checkpoints=True,
            state_dir=state_dir,
        )

        # Load schema if saved
        if checkpoint.schema_hash:
            schema_file = Path(state_dir) / checkpoint.pipeline_id / "schema.yaml"
            if schema_file.exists():
                pipeline._schema = Schema.load(schema_file)
                pipeline._is_fitted = True

        return pipeline

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "Pipeline":
        """
        Load a pipeline from a specific checkpoint directory.

        Args:
            checkpoint_dir: Path to checkpoint directory

        Returns:
            Loaded pipeline
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_manager = CheckpointManager(checkpoint_dir.parent)
        checkpoint = checkpoint_manager.load(checkpoint_dir)

        pipeline = cls(
            enable_checkpoints=True,
            state_dir=str(checkpoint_dir.parent),
        )

        # Load schema
        schema_file = checkpoint_dir / "schema.yaml"
        if schema_file.exists():
            pipeline._schema = Schema.load(schema_file)
            pipeline._is_fitted = True

        return pipeline

    def fit(
        self,
        data: str | Path,
        schema: Schema | None = None,
        **kwargs,
    ) -> "Pipeline":
        """
        Fit the pipeline (split PDFs and detect/set schema).

        The flow is:
        1. Split PDFs into chunks (if auto_split enabled)
        2. Detect schema from 10% of chunks (max 10 files)

        Args:
            data: Path to input documents
            schema: Optional schema to use

        Returns:
            self
        """
        input_path = Path(data)

        # Initialize components
        self._initialize_components(input_path)

        # Step 1: Split PDFs first (before schema detection)
        if self.auto_split:
            Logger.log_stage("PDF Splitting", 1, 3)
            self._split_pdfs(input_path)

        # Step 2: Use provided schema or detect from split chunks
        if schema is not None:
            self._schema = schema
        elif self._schema is None and self.auto_detect_schema:
            Logger.log_stage("Schema Detection", 2, 3)
            detector = SchemaDetector(
                provider=self._get_detection_provider(),
                detection_mode=self.detection_mode,
                purpose=self.purpose,
                seed=self.seed,
            )
            # Now schema detection runs on the split chunks
            detector.fit(str(input_path), tracker=self._tracker)
            self._schema = detector.schema

            # Save schema for resume
            if self.enable_checkpoints and self._state_manager:
                schema_file = (
                    Path(self.state_dir)
                    / self._state_manager.checkpoint.pipeline_id
                    / "schema.yaml"
                )
                self._schema.save(schema_file)
                self._state_manager.checkpoint_manager.set_schema_hash(
                    self._schema.compute_hash()
                )

        self._is_fitted = True
        return self

    def transform(
        self,
        data: str | Path,
        force_restart: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Transform documents (extract data).

        Note: PDF splitting is now done in fit(), not transform().

        Args:
            data: Path to input documents
            force_restart: If True, ignore checkpoints and start fresh

        Returns:
            DataFrame with extracted records
        """
        if not self._is_fitted:
            raise ValueError("Pipeline must be fit before transform. Call fit() first.")

        input_path = Path(data)

        # Initialize if not already done
        if self._state_manager is None:
            self._initialize_components(input_path)

        # Clear checkpoints if force restart
        if force_restart and self._state_manager:
            self._state_manager.clear_checkpoints()

        # Extract data (PDFs already split in fit())
        Logger.log_stage("Data Extraction", 3, 3)
        extraction_provider = self._get_extraction_provider()
        extractor = LLMExtractor(
            schema=self._schema,
            provider=extraction_provider,
            deduplicate=self.deduplicate,
            sample_ratio=self.extraction_sample_ratio,
            max_samples=self.extraction_max_samples,
            seed=self.seed,
        )
        extractor._schema = self._schema
        extractor._is_fitted = True
        extractor.provider = extraction_provider
        extractor._prompt_generator = extractor._prompt_generator or \
            __import__('structify.extractors.prompt_generator', fromlist=['PromptGenerator']).PromptGenerator(self._schema)
        extractor._validator = extractor._validator or \
            __import__('structify.extractors.validator', fromlist=['ResponseValidator']).ResponseValidator(self._schema)

        self._results = extractor.transform(
            str(input_path),
            tracker=self._tracker,
            checkpoint_manager=self._state_manager.checkpoint_manager if self._state_manager else None,
        )

        # Cleanup
        if self._tracker:
            self._tracker.finish()

        if self._state_manager:
            self._state_manager.stop()

        return self._results

    def fit_transform(
        self,
        data: str | Path,
        schema: Schema | None = None,
        force_restart: bool = False,
        selected_fields: list[str] | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            data: Path to input documents
            schema: Optional schema to use
            force_restart: If True, ignore checkpoints
            selected_fields: Optional list of field names to keep after
                schema detection. Use this to filter fields in one call.

        Returns:
            DataFrame with extracted records
        """
        self.fit(data, schema=schema)

        # Apply field selection if provided
        if selected_fields:
            self.select_fields(selected_fields)

        return self.transform(data, force_restart=force_restart)

    def get_schema_dict(self) -> dict[str, str]:
        """
        Get detected schema as {field_name: description} dict.

        Use this to review the detected schema before extraction.
        Then call select_fields() with the field names you want to keep.

        Returns:
            Dict mapping field names to their descriptions

        Raises:
            SchemaError: If pipeline not fitted yet

        Example:
            >>> pipeline.fit("papers/")
            >>> schema = pipeline.get_schema_dict()
            >>> print(schema)
            {'estimate_value': 'The coefficient...', 'methodology': '...'}
        """
        if self._schema is None:
            from structify.core.exceptions import SchemaError
            raise SchemaError("No schema detected. Call fit() first.")
        return SchemaReviewer.to_dict(self._schema)

    def select_fields(self, field_names: list[str]) -> "Pipeline":
        """
        Filter schema to only include specified fields.

        Call this after fit() and before transform() to select which
        fields to extract.

        Args:
            field_names: List of field names to keep

        Returns:
            self (for chaining)

        Raises:
            SchemaError: If pipeline not fitted or invalid field names

        Example:
            >>> pipeline.fit("papers/")
            >>> pipeline.select_fields(["estimate_value", "methodology"])
            >>> results = pipeline.transform("papers/")
        """
        if self._schema is None:
            from structify.core.exceptions import SchemaError
            raise SchemaError("No schema detected. Call fit() first.")
        self._schema = SchemaReviewer.select_fields(self._schema, field_names)
        return self

    def save_schema(self, path: str | Path) -> None:
        """
        Save detected schema to file for later reuse.

        Allows resuming from the extraction step without re-running
        schema detection. Supports JSON and YAML formats.

        Args:
            path: Output path (use .json or .yaml extension)

        Raises:
            ValueError: If no schema to save

        Example:
            >>> pipeline.fit(data_dir)
            >>> pipeline.save_schema("my_schema.json")
            >>> # Later:
            >>> pipeline = Pipeline(schema="my_schema.json")
        """
        if self._schema is None:
            raise ValueError("No schema to save. Call fit() first.")
        self._schema.save(path)
        logger.info(f"Schema saved to {path}")

    def _initialize_components(self, input_path: Path) -> None:
        """Initialize pipeline components."""
        # Initialize provider
        if self.provider is None:
            # Get provider from config
            provider_name = Config.get("default_provider") or "gemini"
            self.provider = get_provider(provider_name)
        elif isinstance(self.provider, str):
            # Provider name passed as string
            self.provider = get_provider(self.provider)
        self.provider.ensure_initialized()

        # Initialize state manager
        if self.enable_checkpoints:
            self._state_manager = StateManager(
                state_dir=self.state_dir,
                enable_checkpoints=True,
            )
            self._state_manager.start(input_path)

        # Initialize progress tracker
        self._tracker = ProgressTracker()

    def _get_detection_provider(self) -> BaseLLMProvider:
        """Get provider for schema detection (uses detection_model if set)."""
        if self.detection_model:
            # Get provider type from config or current provider
            provider_name = Config.get("default_provider") or "gemini"
            provider = get_provider(provider_name, model=self.detection_model)
            provider.ensure_initialized()
            return provider
        return self.provider

    def _get_extraction_provider(self) -> BaseLLMProvider:
        """Get provider for data extraction (uses extraction_model if set)."""
        if self.extraction_model:
            # Get provider type from config or current provider
            provider_name = Config.get("default_provider") or "gemini"
            provider = get_provider(provider_name, model=self.extraction_model)
            provider.ensure_initialized()
            return provider
        return self.provider

    def _split_pdfs(self, input_path: Path) -> None:
        """Split PDFs if needed."""
        # Check if already split
        if self._state_manager and self._state_manager.is_stage_completed("split"):
            logger.info("PDF splitting already completed, skipping")
            return

        # Check if input is already split chunks
        loader = PDFLoader()
        docs = loader.load_directory(input_path)

        # If any document has multiple chunks, assume already split
        if any(doc.total_chunks > 1 for doc in docs):
            logger.info("Documents appear to already be split")
            if self._state_manager:
                self._state_manager.complete_stage("split")
            return

        # Split PDFs
        splitter = PDFSplitter(pages_per_chunk=self.pages_per_chunk)
        splitter.fit(input_path)
        splitter.transform(input_path, tracker=self._tracker)

        if self._state_manager:
            self._state_manager.complete_stage("split")

    def set_params(self, **params) -> "Pipeline":
        """
        Set pipeline parameters using sklearn-style notation.

        Example:
            pipeline.set_params(
                pages_per_chunk=5,
                deduplicate=False,
            )

        Args:
            **params: Parameter key-value pairs

        Returns:
            self
        """
        for key, value in params.items():
            # Handle nested params (e.g., "split__pages_per_chunk")
            if "__" in key:
                step_name, param_name = key.split("__", 1)
                for name, step in self.steps:
                    if name == step_name:
                        step.set_params(**{param_name: value})
                        break
            elif hasattr(self, key):
                setattr(self, key, value)

        return self

    def get_params(self) -> dict[str, Any]:
        """Get all pipeline parameters."""
        return {
            "pages_per_chunk": self.pages_per_chunk,
            "auto_split": self.auto_split,
            "auto_detect_schema": self.auto_detect_schema,
            "deduplicate": self.deduplicate,
            "enable_checkpoints": self.enable_checkpoints,
            "state_dir": self.state_dir,
            "detection_mode": self.detection_mode,
            "purpose": self.purpose,
            "seed": self.seed,
            "detection_model": self.detection_model,
            "extraction_model": self.extraction_model,
            "extraction_sample_ratio": self.extraction_sample_ratio,
            "extraction_max_samples": self.extraction_max_samples,
        }

    def save(self, path: str | Path) -> None:
        """
        Save the pipeline configuration.

        Args:
            path: Path to save the pipeline
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "params": self.get_params(),
            "schema": self._schema.to_dict() if self._schema else None,
        }

        with open(path, "wb") as f:
            pickle.dump(save_data, f)

        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        """
        Load a saved pipeline.

        Args:
            path: Path to the saved pipeline

        Returns:
            Loaded pipeline
        """
        path = Path(path)

        with open(path, "rb") as f:
            save_data = pickle.load(f)

        pipeline = cls(**save_data["params"])

        if save_data.get("schema"):
            pipeline._schema = Schema.from_dict(save_data["schema"])
            pipeline._is_fitted = True

        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    def save_checkpoint(self) -> None:
        """Force save the current checkpoint."""
        if self._state_manager:
            self._state_manager.save_checkpoint()

    def clear_checkpoints(self) -> None:
        """Clear all checkpoints."""
        if self._state_manager:
            self._state_manager.clear_checkpoints()

    @property
    def schema(self) -> Schema | None:
        """Get the current schema."""
        return self._schema

    @property
    def results(self) -> pd.DataFrame | None:
        """Get the last extraction results."""
        return self._results

    @property
    def is_fitted(self) -> bool:
        """Check if pipeline is fitted."""
        return self._is_fitted
