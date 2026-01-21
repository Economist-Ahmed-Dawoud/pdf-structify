"""PDF splitting utility."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

from pypdf import PdfReader, PdfWriter

from structify.core.base import BaseTransformer
from structify.utils.logging import get_logger, Logger
from structify.progress.tracker import ProgressTracker

logger = get_logger("splitter")


@dataclass
class SplitResult:
    """Result of a PDF split operation."""

    source_file: Path
    output_dir: Path
    chunks: list[Path] = field(default_factory=list)
    total_pages: int = 0
    pages_per_chunk: int = 10


def sanitize_filename(name: str, max_length: int = 50) -> str:
    """
    Sanitize a filename for safe use.

    Args:
        name: Original filename
        max_length: Maximum length

    Returns:
        Sanitized filename
    """
    # Remove or replace problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', "_", name)
    safe_name = re.sub(r"\s+", "_", safe_name)
    safe_name = re.sub(r"_+", "_", safe_name)
    safe_name = safe_name.strip("_")

    # Truncate if needed
    if len(safe_name) > max_length:
        safe_name = safe_name[:max_length].rstrip("_")

    return safe_name


class PDFSplitter(BaseTransformer[str | Path, list[SplitResult]]):
    """
    Split PDFs into smaller chunks.

    Sklearn-like API for splitting PDF files into manageable chunks
    for processing with LLMs.
    """

    def __init__(
        self,
        pages_per_chunk: int = 10,
        output_format: str = "{name}_pages_{start}-{end}.pdf",
        create_index: bool = True,
        create_folders: bool = True,
        max_filename_length: int = 50,
    ):
        """
        Initialize the PDF splitter.

        Args:
            pages_per_chunk: Number of pages per chunk
            output_format: Format string for output filenames
            create_index: Whether to create an INDEX.txt file
            create_folders: Whether to create a folder per PDF
            max_filename_length: Maximum filename length
        """
        super().__init__(
            pages_per_chunk=pages_per_chunk,
            output_format=output_format,
            create_index=create_index,
            create_folders=create_folders,
            max_filename_length=max_filename_length,
        )
        self.pages_per_chunk = pages_per_chunk
        self.output_format = output_format
        self.create_index = create_index
        self.create_folders = create_folders
        self.max_filename_length = max_filename_length

    def fit(self, data: str | Path, **kwargs) -> "PDFSplitter":
        """
        Fit is a no-op for PDFSplitter.

        Args:
            data: Input path (ignored)

        Returns:
            self
        """
        self._is_fitted = True
        return self

    def transform(
        self,
        data: str | Path,
        output_path: str | Path | None = None,
        tracker: ProgressTracker | None = None,
        **kwargs,
    ) -> list[SplitResult]:
        """
        Split PDFs from input path.

        Args:
            data: Input path (file or directory)
            output_path: Output directory (defaults to input directory)
            tracker: Optional progress tracker

        Returns:
            List of SplitResult objects
        """
        input_path = Path(data)

        if output_path is None:
            output_path = input_path if input_path.is_dir() else input_path.parent
        output_path = Path(output_path)

        # Find PDF files
        if input_path.is_file():
            pdf_files = [input_path]
        else:
            pdf_files = list(input_path.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {input_path}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to split")

        # Set up progress tracking
        if tracker:
            tracker.add_stage("split", len(pdf_files))
            tracker.start_stage("split")

        results = []
        index_entries = []

        for pdf_file in pdf_files:
            try:
                result = self.split_file(pdf_file, output_path, tracker)
                results.append(result)

                # Add to index
                for chunk in result.chunks:
                    index_entries.append(f"{pdf_file.name} -> {chunk.name}")

                if tracker:
                    tracker.increment(records=len(result.chunks))

            except Exception as e:
                logger.error(f"Error splitting {pdf_file.name}: {e}")
                if tracker:
                    tracker.error_stage(error=str(e))

        # Create index file
        if self.create_index and index_entries:
            index_file = output_path / "INDEX.txt"
            with open(index_file, "w", encoding="utf-8") as f:
                f.write("PDF Split Index\n")
                f.write("=" * 50 + "\n\n")
                for entry in index_entries:
                    f.write(f"{entry}\n")
            logger.info(f"Created index file: {index_file}")

        if tracker:
            tracker.complete_stage("split")

        return results

    def split_file(
        self,
        pdf_path: str | Path,
        output_dir: str | Path,
        tracker: ProgressTracker | None = None,
    ) -> SplitResult:
        """
        Split a single PDF file into chunks.

        Args:
            pdf_path: Path to the PDF file
            output_dir: Output directory
            tracker: Optional progress tracker

        Returns:
            SplitResult object
        """
        pdf_path = Path(pdf_path)
        output_dir = Path(output_dir)

        # Read PDF
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        # Create safe name
        safe_name = sanitize_filename(pdf_path.stem, self.max_filename_length)

        # Create output folder if needed
        if self.create_folders:
            chunk_dir = output_dir / safe_name
            chunk_dir.mkdir(parents=True, exist_ok=True)
        else:
            chunk_dir = output_dir

        result = SplitResult(
            source_file=pdf_path,
            output_dir=chunk_dir,
            total_pages=total_pages,
            pages_per_chunk=self.pages_per_chunk,
        )

        # Split into chunks
        chunk_num = 0
        for start_page in range(0, total_pages, self.pages_per_chunk):
            end_page = min(start_page + self.pages_per_chunk, total_pages)

            # Create writer for this chunk
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Generate filename (1-indexed for human readability)
            chunk_filename = self.output_format.format(
                name=safe_name,
                start=start_page + 1,
                end=end_page,
            )
            chunk_path = chunk_dir / chunk_filename

            # Write chunk
            with open(chunk_path, "wb") as f:
                writer.write(f)

            result.chunks.append(chunk_path)
            chunk_num += 1

            if tracker:
                tracker.log_substep(
                    f"Created {chunk_filename} (pages {start_page + 1}-{end_page})",
                    style="info",
                )

        logger.info(f"Split {pdf_path.name}: {total_pages} pages -> {len(result.chunks)} chunks")

        return result

    def iter_chunks(self, pdf_path: str | Path) -> Iterator[tuple[int, int, bytes]]:
        """
        Iterate over chunks of a PDF without writing files.

        Yields:
            Tuple of (start_page, end_page, pdf_bytes)
        """
        pdf_path = Path(pdf_path)
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)

        for start_page in range(0, total_pages, self.pages_per_chunk):
            end_page = min(start_page + self.pages_per_chunk, total_pages)

            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])

            # Get bytes
            import io
            buffer = io.BytesIO()
            writer.write(buffer)
            buffer.seek(0)

            yield (start_page + 1, end_page, buffer.read())
