"""Output writers for saving extraction results."""

from pathlib import Path
from typing import Any
from datetime import datetime

import pandas as pd

from structify.utils.logging import get_logger

logger = get_logger("output")


class OutputWriter:
    """
    Utility class for saving extraction results in various formats.

    Supports:
    - CSV
    - JSON
    - Parquet
    - Excel
    """

    @staticmethod
    def to_csv(
        df: pd.DataFrame,
        path: str | Path,
        index: bool = False,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to CSV.

        Args:
            df: DataFrame to save
            path: Output path
            index: Whether to include index
            **kwargs: Additional pandas to_csv arguments

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(path, index=index, **kwargs)
        logger.info(f"Saved {len(df)} records to {path}")

        return path

    @staticmethod
    def to_json(
        df: pd.DataFrame,
        path: str | Path,
        orient: str = "records",
        indent: int = 2,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to JSON.

        Args:
            df: DataFrame to save
            path: Output path
            orient: JSON orientation (records, columns, etc.)
            indent: Indentation level
            **kwargs: Additional pandas to_json arguments

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_json(path, orient=orient, indent=indent, **kwargs)
        logger.info(f"Saved {len(df)} records to {path}")

        return path

    @staticmethod
    def to_parquet(
        df: pd.DataFrame,
        path: str | Path,
        index: bool = False,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to Parquet.

        Args:
            df: DataFrame to save
            path: Output path
            index: Whether to include index
            **kwargs: Additional pandas to_parquet arguments

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(path, index=index, **kwargs)
        logger.info(f"Saved {len(df)} records to {path}")

        return path

    @staticmethod
    def to_excel(
        df: pd.DataFrame,
        path: str | Path,
        sheet_name: str = "Data",
        index: bool = False,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame to Excel.

        Args:
            df: DataFrame to save
            path: Output path
            sheet_name: Name of the worksheet
            index: Whether to include index
            **kwargs: Additional pandas to_excel arguments

        Returns:
            Path to saved file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_excel(path, sheet_name=sheet_name, index=index, **kwargs)
        logger.info(f"Saved {len(df)} records to {path}")

        return path

    @staticmethod
    def save(
        df: pd.DataFrame,
        path: str | Path,
        format: str | None = None,
        **kwargs,
    ) -> Path:
        """
        Save DataFrame, auto-detecting format from extension.

        Args:
            df: DataFrame to save
            path: Output path
            format: Optional format override (csv, json, parquet, excel)
            **kwargs: Additional arguments for the specific writer

        Returns:
            Path to saved file
        """
        path = Path(path)

        # Detect format from extension if not provided
        if format is None:
            ext = path.suffix.lower()
            format_map = {
                ".csv": "csv",
                ".json": "json",
                ".parquet": "parquet",
                ".xlsx": "excel",
                ".xls": "excel",
            }
            format = format_map.get(ext, "csv")

        # Save in appropriate format
        if format == "csv":
            return OutputWriter.to_csv(df, path, **kwargs)
        elif format == "json":
            return OutputWriter.to_json(df, path, **kwargs)
        elif format == "parquet":
            return OutputWriter.to_parquet(df, path, **kwargs)
        elif format == "excel":
            return OutputWriter.to_excel(df, path, **kwargs)
        else:
            raise ValueError(f"Unknown format: {format}")

    @staticmethod
    def save_with_metadata(
        df: pd.DataFrame,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
        format: str = "csv",
    ) -> tuple[Path, Path]:
        """
        Save DataFrame with accompanying metadata file.

        Args:
            df: DataFrame to save
            path: Output path
            metadata: Metadata to save
            format: Output format

        Returns:
            Tuple of (data_path, metadata_path)
        """
        import json

        path = Path(path)
        data_path = OutputWriter.save(df, path, format=format)

        # Create metadata
        meta = metadata or {}
        meta.update({
            "records": len(df),
            "columns": list(df.columns),
            "created_at": datetime.now().isoformat(),
        })

        # Save metadata
        meta_path = path.with_suffix(".meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Saved metadata to {meta_path}")

        return data_path, meta_path
