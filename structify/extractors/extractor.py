"""Main extraction classes for structify."""

import random
import time
from pathlib import Path
from typing import Any

import pandas as pd

from structify.core.base import BaseExtractor
from structify.schema.types import Schema
from structify.schema.detector import SchemaDetector
from structify.providers.gemini import GeminiProvider
from structify.preprocessing.loader import PDFLoader, PDFChunk
from structify.extractors.prompt_generator import PromptGenerator
from structify.extractors.validator import ResponseValidator
from structify.progress.tracker import ProgressTracker
from structify.progress.checkpoint import CheckpointManager
from structify.utils.logging import get_logger, Logger

logger = get_logger("extractor")


class LLMExtractor(BaseExtractor[str, pd.DataFrame]):
    """
    Extract structured data from documents using LLMs.

    Sklearn-like API for document data extraction:
    - fit(): Detect or set schema
    - transform(): Extract data using schema
    - fit_transform(): Detect schema and extract in one step
    """

    def __init__(
        self,
        schema: Schema | None = None,
        provider: GeminiProvider | None = None,
        deduplicate: bool = True,
        dedup_fields: list[str] | None = None,
        between_calls_delay: int = 3,
        sample_ratio: float | None = None,
        max_samples: int | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the extractor.

        Args:
            schema: Schema defining fields to extract (optional, can be detected)
            provider: LLM provider for extraction
            deduplicate: Whether to deduplicate results
            dedup_fields: Fields to use for deduplication
            between_calls_delay: Delay between API calls in seconds
            sample_ratio: Fraction of documents to sample (0.0-1.0), None for all
            max_samples: Maximum number of documents to process, None for all
            seed: Random seed for reproducible sampling
        """
        super().__init__(
            schema=schema,
            deduplicate=deduplicate,
            dedup_fields=dedup_fields,
            between_calls_delay=between_calls_delay,
        )

        self._schema = schema
        self.provider = provider
        self.deduplicate = deduplicate
        self.dedup_fields = dedup_fields
        self.between_calls_delay = between_calls_delay
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples
        self.seed = seed

        self._prompt_generator: PromptGenerator | None = None
        self._validator: ResponseValidator | None = None

    def fit(
        self,
        data: str,
        schema: Schema | None = None,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> "LLMExtractor":
        """
        Fit the extractor (set or detect schema).

        Args:
            data: Path to documents
            schema: Schema to use (if not provided, will be detected)
            tracker: Optional progress tracker

        Returns:
            self
        """
        if schema is not None:
            self._schema = schema
        elif self._schema is None:
            # Auto-detect schema
            logger.info("No schema provided, detecting from documents...")
            detector = SchemaDetector(provider=self.provider)
            detector.fit(data, tracker=tracker)
            self._schema = detector.schema

        # Initialize components
        self._prompt_generator = PromptGenerator(self._schema)
        self._validator = ResponseValidator(self._schema)

        self._is_fitted = True
        return self

    def transform(
        self,
        data: str,
        tracker: ProgressTracker | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Extract data from documents.

        Args:
            data: Path to documents
            tracker: Optional progress tracker
            checkpoint_manager: Optional checkpoint manager for resume

        Returns:
            DataFrame with extracted records
        """
        if not self._is_fitted:
            raise ValueError("Extractor must be fit before transform")

        # Initialize provider if needed
        if self.provider is None:
            self.provider = GeminiProvider()
        self.provider.ensure_initialized()

        # Load documents
        loader = PDFLoader()
        all_chunks = loader.get_all_chunks(data)

        if not all_chunks:
            logger.warning(f"No documents found in {data}")
            return pd.DataFrame()

        # Apply sampling if configured
        original_count = len(all_chunks)
        all_chunks = self._sample_chunks(all_chunks)
        if len(all_chunks) < original_count:
            logger.info(f"Sampling enabled: {len(all_chunks)}/{original_count} documents selected")

        # Check for resume
        pending_chunks = all_chunks
        existing_records = []

        if checkpoint_manager:
            stage = checkpoint_manager.get_stage("extract")
            if stage and stage.last_completed_item:
                # Load previously extracted records
                existing_records = checkpoint_manager.load_partial_results()

                # Get pending items
                chunk_names = [c.path.name for c in all_chunks]
                pending_names = checkpoint_manager.get_pending_items("extract", chunk_names)

                # Filter chunks
                pending_chunks = [c for c in all_chunks if c.path.name in pending_names]

                logger.info(
                    f"Resuming extraction: {len(all_chunks) - len(pending_chunks)} "
                    f"already processed ({len(existing_records)} records), "
                    f"{len(pending_chunks)} remaining"
                )

        # Set up progress tracking
        if tracker:
            tracker.add_stage("extract", len(all_chunks))
            tracker.start_stage("extract")
            # Update for already processed
            if len(pending_chunks) < len(all_chunks):
                tracker.update(completed=len(all_chunks) - len(pending_chunks))

        # Extract from each chunk
        all_records = existing_records.copy()

        for chunk in pending_chunks:
            try:
                records = self._extract_chunk(chunk, tracker)
                all_records.extend(records)

                # Update checkpoint AND save partial results
                if checkpoint_manager:
                    chunk_idx = pending_chunks.index(chunk)
                    completed = len(all_chunks) - len(pending_chunks) + chunk_idx + 1
                    checkpoint_manager.update_stage(
                        "extract",
                        completed_items=completed,
                        last_completed_item=chunk.path.name,
                        records_extracted=len(all_records),
                    )
                    # Save records incrementally - survives shutdown
                    checkpoint_manager.save_partial_results(all_records)

                # Delay between calls
                time.sleep(self.between_calls_delay)

            except KeyboardInterrupt:
                logger.warning("Extraction interrupted - saving progress")
                # Save current records before exiting
                if checkpoint_manager and all_records:
                    checkpoint_manager.save_partial_results(all_records)
                break
            except Exception as e:
                logger.error(f"Error extracting from {chunk.name}: {e}")
                if tracker:
                    tracker.log_substep(f"Error: {e}", style="error")

        # Complete tracking
        if tracker:
            tracker.complete_stage("extract")

        if checkpoint_manager:
            checkpoint_manager.complete_stage("extract", len(all_records))

        # Deduplicate
        if self.deduplicate and all_records:
            all_records = self._deduplicate(all_records)

        # Convert to DataFrame
        df = pd.DataFrame(all_records)

        logger.info(f"Extracted {len(df)} records from {len(all_chunks)} documents")
        return df

    def _extract_chunk(
        self,
        chunk: PDFChunk,
        tracker: ProgressTracker | None = None,
    ) -> list[dict[str, Any]]:
        """Extract data from a single chunk."""
        logger.debug(f"Extracting from: {chunk.name}")

        if tracker:
            tracker.update(current_item=chunk.name)

        # Build prompt
        prompt = self._prompt_generator.with_document_info(
            document_name=chunk.source_pdf,
            part_num=chunk.chunk_index + 1,
            total_parts=chunk.total_chunks,
        )

        # Upload file and generate
        file_ref = self.provider.upload_file(str(chunk.path))
        response = self.provider.generate(prompt, file_ref)

        # Validate response
        records = self._validator.validate(response)

        if tracker:
            tracker.increment(records=len(records))

        return records

    def _deduplicate(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate records."""
        if not records:
            return records

        # Determine dedup fields
        if self.dedup_fields:
            key_fields = self.dedup_fields
        else:
            # Use all required fields + first few optional
            key_fields = [f.name for f in self._schema.get_required_fields()]
            if not key_fields:
                key_fields = self._schema.get_field_names()[:3]

        seen = set()
        unique = []

        for record in records:
            key = tuple(str(record.get(f, "")) for f in key_fields)
            if key not in seen:
                seen.add(key)
                unique.append(record)

        if len(unique) < len(records):
            logger.info(f"Deduplicated: {len(records)} -> {len(unique)} records")

        return unique

    def _sample_chunks(self, chunks: list[PDFChunk]) -> list[PDFChunk]:
        """
        Sample chunks for extraction if sampling is enabled.

        Args:
            chunks: All available chunks

        Returns:
            Sampled chunks if sampling enabled, otherwise all chunks
        """
        if self.sample_ratio is None and self.max_samples is None:
            return chunks  # No sampling

        total = len(chunks)

        if self.sample_ratio is not None:
            sample_size = int(total * self.sample_ratio)
        else:
            sample_size = total

        if self.max_samples is not None:
            sample_size = min(sample_size, self.max_samples)

        # Ensure at least 1 sample, at most total
        sample_size = max(1, min(sample_size, total))

        if sample_size >= total:
            return chunks  # No sampling needed

        if self.seed is not None:
            rng = random.Random(self.seed)
            logger.debug(f"Using seed {self.seed} for reproducible sampling")
            return rng.sample(chunks, sample_size)

        return random.sample(chunks, sample_size)

    def save(self, df: pd.DataFrame, path: str | Path, format: str = "csv") -> None:
        """
        Save extraction results.

        Args:
            df: DataFrame to save
            path: Output path
            format: Output format (csv, json, parquet)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "csv":
            df.to_csv(path, index=False)
        elif format == "json":
            df.to_json(path, orient="records", indent=2)
        elif format == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")

        logger.info(f"Saved {len(df)} records to {path}")


class MultiExtractor:
    """
    Extract multiple data types from the same documents.

    Useful when you need to extract different kinds of information
    (e.g., impact estimates AND manufacturer benefits) from the same PDFs.
    """

    def __init__(
        self,
        schemas: dict[str, Schema],
        provider: GeminiProvider | None = None,
        deduplicate: bool = True,
    ):
        """
        Initialize the multi-extractor.

        Args:
            schemas: Dictionary mapping names to schemas
            provider: Shared LLM provider
            deduplicate: Whether to deduplicate results
        """
        self.schemas = schemas
        self.provider = provider or GeminiProvider()
        self.deduplicate = deduplicate

        self._extractors: dict[str, LLMExtractor] = {}

    def fit(self, data: str, **kwargs) -> "MultiExtractor":
        """Fit all extractors."""
        for name, schema in self.schemas.items():
            extractor = LLMExtractor(
                schema=schema,
                provider=self.provider,
                deduplicate=self.deduplicate,
            )
            extractor.fit(data, schema=schema)
            self._extractors[name] = extractor

        return self

    def transform(
        self,
        data: str,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """
        Extract all data types from documents.

        Args:
            data: Path to documents
            tracker: Optional progress tracker

        Returns:
            Dictionary mapping names to DataFrames
        """
        results = {}

        for name, extractor in self._extractors.items():
            logger.info(f"Extracting: {name}")
            results[name] = extractor.transform(data, tracker=tracker)

        return results

    def fit_transform(
        self,
        data: str,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Fit and transform in one step."""
        return self.fit(data).transform(data, tracker=tracker)
