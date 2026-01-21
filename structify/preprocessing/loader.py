"""PDF loading utilities."""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Iterator

from structify.utils.logging import get_logger

logger = get_logger("loader")


@dataclass
class PDFChunk:
    """Represents a single PDF chunk."""

    path: Path
    source_pdf: str
    start_page: int
    end_page: int
    chunk_index: int
    total_chunks: int

    @property
    def name(self) -> str:
        """Get the chunk filename."""
        return self.path.name

    @property
    def page_range(self) -> str:
        """Get the page range string."""
        return f"{self.start_page}-{self.end_page}"

    def __str__(self) -> str:
        return f"{self.source_pdf} [{self.page_range}]"


@dataclass
class PDFDocument:
    """Represents a PDF document with its chunks."""

    name: str
    source_path: Path | None
    chunks: list[PDFChunk] = field(default_factory=list)

    @property
    def total_chunks(self) -> int:
        """Get total number of chunks."""
        return len(self.chunks)

    def __iter__(self) -> Iterator[PDFChunk]:
        """Iterate over chunks."""
        return iter(self.chunks)


class PDFLoader:
    """
    Load PDF files and chunks from directories.

    Handles both original PDFs and pre-split chunk directories.
    """

    def __init__(self, chunk_pattern: str = r"_pages_(\d+)-(\d+)\.pdf$"):
        """
        Initialize the loader.

        Args:
            chunk_pattern: Regex pattern to extract page numbers from chunk filenames
        """
        self.chunk_pattern = re.compile(chunk_pattern)

    def load_directory(self, path: str | Path) -> list[PDFDocument]:
        """
        Load all PDF documents from a directory.

        Automatically detects if directory contains:
        - Original PDF files
        - Subdirectories with split chunks
        - Mixed content

        Args:
            path: Directory path

        Returns:
            List of PDFDocument objects
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        documents = []

        # Check for subdirectories (split chunks)
        subdirs = [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]

        if subdirs:
            # Load from subdirectories
            for subdir in sorted(subdirs):
                doc = self._load_chunk_directory(subdir)
                if doc and doc.chunks:
                    documents.append(doc)

        # Also check for PDF files directly in the directory
        pdf_files = list(path.glob("*.pdf"))
        for pdf_file in sorted(pdf_files):
            # Skip if this looks like a chunk file
            if self.chunk_pattern.search(pdf_file.name):
                continue

            doc = PDFDocument(
                name=pdf_file.stem,
                source_path=pdf_file,
                chunks=[
                    PDFChunk(
                        path=pdf_file,
                        source_pdf=pdf_file.name,
                        start_page=1,
                        end_page=self._get_page_count(pdf_file),
                        chunk_index=0,
                        total_chunks=1,
                    )
                ],
            )
            documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents from {path}")
        return documents

    def _load_chunk_directory(self, path: Path) -> PDFDocument | None:
        """
        Load chunks from a directory.

        Args:
            path: Directory containing chunks

        Returns:
            PDFDocument or None
        """
        chunks = []
        pdf_files = sorted(path.glob("*.pdf"), key=self._extract_page_num)

        for i, pdf_file in enumerate(pdf_files):
            match = self.chunk_pattern.search(pdf_file.name)
            if match:
                start_page = int(match.group(1))
                end_page = int(match.group(2))
            else:
                # Single file without page numbers
                start_page = 1
                end_page = self._get_page_count(pdf_file)

            chunk = PDFChunk(
                path=pdf_file,
                source_pdf=path.name,
                start_page=start_page,
                end_page=end_page,
                chunk_index=i,
                total_chunks=len(pdf_files),
            )
            chunks.append(chunk)

        if not chunks:
            return None

        # Update total_chunks now that we know the count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return PDFDocument(name=path.name, source_path=path, chunks=chunks)

    def _extract_page_num(self, pdf_path: Path) -> int:
        """Extract starting page number from filename for sorting."""
        match = self.chunk_pattern.search(pdf_path.name)
        if match:
            return int(match.group(1))
        return 0

    def _get_page_count(self, pdf_path: Path) -> int:
        """Get the page count of a PDF file."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            return len(reader.pages)
        except Exception:
            return 0

    def iter_chunks(self, path: str | Path) -> Iterator[PDFChunk]:
        """
        Iterate over all chunks in a directory.

        Args:
            path: Directory path

        Yields:
            PDFChunk objects
        """
        documents = self.load_directory(path)
        for doc in documents:
            for chunk in doc.chunks:
                yield chunk

    def get_all_chunks(self, path: str | Path) -> list[PDFChunk]:
        """
        Get all chunks as a flat list.

        Args:
            path: Directory path

        Returns:
            List of PDFChunk objects
        """
        return list(self.iter_chunks(path))

    def count_documents(self, path: str | Path) -> int:
        """Count documents in a directory."""
        return len(self.load_directory(path))

    def count_chunks(self, path: str | Path) -> int:
        """Count total chunks in a directory."""
        return len(self.get_all_chunks(path))
