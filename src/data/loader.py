"""
Data Loader Module

Handles loading project data from various file formats (CSV, JSON, Excel).
"""

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger


class DataLoader:
    """Load project data from various file formats."""

    SUPPORTED_FORMATS = [".csv", ".json", ".xlsx", ".xls"]

    def __init__(self):
        """Initialize the DataLoader."""
        self.last_loaded_path: Optional[Path] = None
        self.last_loaded_format: Optional[str] = None

    def load(
        self,
        file_path: Union[str, Path],
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from a file.

        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to the underlying loader

        Returns:
            DataFrame containing the loaded data

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

        logger.info(f"Loading data from {file_path}")

        if suffix == ".csv":
            df = self._load_csv(file_path, **kwargs)
        elif suffix == ".json":
            df = self._load_json(file_path, **kwargs)
        elif suffix in [".xlsx", ".xls"]:
            df = self._load_excel(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")

        self.last_loaded_path = file_path
        self.last_loaded_format = suffix

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

        return df

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        default_kwargs = {
            "encoding": "utf-8",
            "parse_dates": ["start_date", "planned_end_date", "actual_end_date"],
        }
        default_kwargs.update(kwargs)

        try:
            return pd.read_csv(file_path, **default_kwargs)
        except Exception as e:
            logger.warning(f"Error with date parsing, retrying without: {e}")
            default_kwargs.pop("parse_dates", None)
            return pd.read_csv(file_path, **default_kwargs)

    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list of records and nested structure
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and "projects" in data:
            return pd.DataFrame(data["projects"])
        else:
            return pd.DataFrame([data])

    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load data from Excel file."""
        sheet_name = kwargs.pop("sheet_name", 0)
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    def load_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from bytes (e.g., from Streamlit file uploader).

        Args:
            file_bytes: Raw file bytes
            file_name: Original filename (used to determine format)
            **kwargs: Additional arguments

        Returns:
            DataFrame containing the loaded data
        """
        from io import BytesIO, StringIO

        suffix = Path(file_name).suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(StringIO(file_bytes.decode("utf-8")), **kwargs)
        elif suffix == ".json":
            data = json.loads(file_bytes.decode("utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict) and "projects" in data:
                return pd.DataFrame(data["projects"])
            return pd.DataFrame([data])
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(BytesIO(file_bytes), **kwargs)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
