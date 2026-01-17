"""
Data Loader Module
==================

This module provides functionality for loading project data from various file formats.

Supports both PRISM native format and processed Apache JIRA data.

Data Sources:
    - PRISM native format (sample_projects.csv)
    - Processed Apache JIRA data (jira_projects.csv)
      Source: https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues

Example:
    >>> from src.data.loader import DataLoader
    >>> loader = DataLoader()
    >>> df = loader.load("data/raw/sample_projects.csv")
"""

import json
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from loguru import logger


class DataLoader:
    """
    Load project data from various file formats.

    This class follows the Single Responsibility Principle by focusing solely
    on data loading operations. It supports CSV, JSON, and Excel formats.

    :cvar SUPPORTED_FORMATS: List of supported file extensions.
    :vartype SUPPORTED_FORMATS: list[str]
    :cvar JIRA_TO_PRISM_MAPPING: Column mapping from JIRA to PRISM format.
    :vartype JIRA_TO_PRISM_MAPPING: dict[str, str]

    :ivar last_loaded_path: Path of the last loaded file.
    :vartype last_loaded_path: Optional[Path]
    :ivar last_loaded_format: Format of the last loaded file.
    :vartype last_loaded_format: Optional[str]
    :ivar data_source: Detected data source ('prism' or 'jira').
    :vartype data_source: Optional[str]

    Example:
        >>> loader = DataLoader()
        >>> df = loader.load("projects.csv")
    """

    SUPPORTED_FORMATS: list[str] = [".csv", ".json", ".xlsx", ".xls"]

    JIRA_TO_PRISM_MAPPING: dict[str, str] = {
        "project_id": "project_id",
        "project_name": "project_name",
        "project_type": "project_type",
        "start_date": "start_date",
        "planned_end_date": "planned_end_date",
        "actual_end_date": "actual_end_date",
        "budget": "budget",
        "spent": "spent",
        "planned_hours": "planned_hours",
        "actual_hours": "actual_hours",
        "team_size": "team_size",
        "completion_rate": "completion_rate",
        "status": "status",
        "priority": "priority",
        "methodology": "methodology",
        "department": "department",
        "client_type": "client_type",
        "complexity_score": "complexity_score",
        "dependencies": "dependencies",
        "velocity": "velocity",
        "defect_rate": "defect_rate",
        "team_turnover": "team_turnover",
        "risk_level": "risk_level",
        "status_comments": "status_comments",
        "project_description": "project_description",
        "team_feedback": "team_feedback",
    }

    def __init__(self) -> None:
        """Initialize the DataLoader with default state."""
        self.last_loaded_path: Optional[Path] = None
        self.last_loaded_format: Optional[str] = None
        self.data_source: Optional[str] = None

    def load(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from a file.

        :param file_path: Path to the data file.
        :type file_path: Union[str, Path]
        :param kwargs: Additional arguments passed to the underlying loader
            (e.g., encoding, sheet_name for Excel).
        :return: DataFrame containing the loaded data.
        :rtype: pd.DataFrame
        :raises ValueError: If file format is not supported.
        :raises FileNotFoundError: If file does not exist.

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load("data/projects.csv", encoding="utf-8")
        """
        file_path = Path(file_path)
        self._validate_file_exists(file_path)
        self._validate_format(file_path)

        logger.info(f"Loading data from {file_path}")

        df = self._load_by_format(file_path, **kwargs)
        self._update_state(file_path)
        df = self._detect_and_normalize_format(df)

        logger.info(
            f"Loaded {len(df)} records with {len(df.columns)} columns "
            f"(source: {self.data_source})"
        )
        return df

    def _validate_file_exists(self, file_path: Path) -> None:
        """
        Validate that the file exists.

        :param file_path: Path to validate.
        :type file_path: Path
        :raises FileNotFoundError: If file does not exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    def _validate_format(self, file_path: Path) -> None:
        """
        Validate that the file format is supported.

        :param file_path: Path to validate.
        :type file_path: Path
        :raises ValueError: If format is not supported.
        """
        suffix = file_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                f"Supported formats: {self.SUPPORTED_FORMATS}"
            )

    def _load_by_format(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data using the appropriate loader for the file format.

        :param file_path: Path to the file.
        :type file_path: Path
        :param kwargs: Additional arguments for the loader.
        :return: Loaded DataFrame.
        :rtype: pd.DataFrame
        """
        suffix = file_path.suffix.lower()
        loaders = {
            ".csv": self._load_csv,
            ".json": self._load_json,
            ".xlsx": self._load_excel,
            ".xls": self._load_excel,
        }
        return loaders[suffix](file_path, **kwargs)

    def _update_state(self, file_path: Path) -> None:
        """
        Update loader state after successful load.

        :param file_path: Path of the loaded file.
        :type file_path: Path
        """
        self.last_loaded_path = file_path
        self.last_loaded_format = file_path.suffix.lower()

    def _detect_and_normalize_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect the data format and normalize to PRISM schema.

        :param df: Raw loaded DataFrame.
        :type df: pd.DataFrame
        :return: Normalized DataFrame.
        :rtype: pd.DataFrame
        """
        jira_indicators = [
            "total_issues",
            "blocker_count",
            "reopen_rate",
            "avg_resolution_days",
        ]
        jira_cols = [col for col in jira_indicators if col in df.columns]

        if len(jira_cols) >= 2:
            self.data_source = "jira"
            logger.info("Detected Apache JIRA data format")
            return self._normalize_jira_data(df)

        self.data_source = "prism"
        logger.info("Detected PRISM native data format")
        return df

    def _normalize_jira_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize JIRA data to be compatible with PRISM processing.

        :param df: JIRA format DataFrame.
        :type df: pd.DataFrame
        :return: Normalized DataFrame.
        :rtype: pd.DataFrame
        """
        df = df.copy()
        df = self._apply_jira_defaults(df)
        df = self._create_synthetic_budget(df)
        df = self._normalize_dates(df)
        return df

    def _apply_jira_defaults(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply default values for missing JIRA columns.

        :param df: DataFrame to update.
        :type df: pd.DataFrame
        :return: DataFrame with defaults applied.
        :rtype: pd.DataFrame
        """
        defaults = {
            "budget": 0,
            "spent": 0,
            "team_turnover": 0.0,
            "team_feedback": "",
            "stakeholder_notes": "",
            "technology_stack": "Java",
        }
        for col, default_value in defaults.items():
            if col not in df.columns:
                df[col] = default_value
            else:
                df[col] = df[col].fillna(default_value)
        return df

    def _create_synthetic_budget(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic budget/spent values if not present.

        :param df: DataFrame to update.
        :type df: pd.DataFrame
        :return: DataFrame with budget values.
        :rtype: pd.DataFrame
        """
        if df["budget"].sum() == 0 and "planned_hours" in df.columns:
            hourly_rate = 100
            df["budget"] = df["planned_hours"] * hourly_rate
            if "actual_hours" in df.columns:
                df["spent"] = df["actual_hours"] * hourly_rate
            else:
                df["spent"] = df["budget"] * 0.8
        return df

    def _normalize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize date columns to datetime format.

        :param df: DataFrame to update.
        :type df: pd.DataFrame
        :return: DataFrame with normalized dates.
        :rtype: pd.DataFrame
        """
        date_cols = ["start_date", "planned_end_date", "actual_end_date"]
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

        # Create planned_end_date if missing
        if "planned_end_date" not in df.columns or df["planned_end_date"].isna().all():
            if "start_date" in df.columns and "project_duration_days" in df.columns:
                df["planned_end_date"] = df["start_date"] + pd.to_timedelta(
                    df["project_duration_days"], unit="D"
                )
        return df

    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.

        :param file_path: Path to CSV file.
        :type file_path: Path
        :param kwargs: Additional arguments for pd.read_csv.
        :return: Loaded DataFrame.
        :rtype: pd.DataFrame
        """
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
        """
        Load data from JSON file.

        :param file_path: Path to JSON file.
        :type file_path: Path
        :param kwargs: Additional arguments (unused, for interface consistency).
        :return: Loaded DataFrame.
        :rtype: pd.DataFrame
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "projects" in data:
            return pd.DataFrame(data["projects"])
        return pd.DataFrame([data])

    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from Excel file.

        :param file_path: Path to Excel file.
        :type file_path: Path
        :param kwargs: Additional arguments for pd.read_excel.
        :return: Loaded DataFrame.
        :rtype: pd.DataFrame
        """
        sheet_name = kwargs.pop("sheet_name", 0)
        return pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)

    def load_jira_data(
        self,
        processed_dir: Optional[Union[str, Path]] = None,
        filename: str = "jira_projects.csv",
    ) -> pd.DataFrame:
        """
        Load processed JIRA data.

        This is a convenience method for loading the output of the
        JIRA preprocessing script.

        :param processed_dir: Directory containing processed data.
            Defaults to data/processed.
        :type processed_dir: Optional[Union[str, Path]]
        :param filename: Name of the processed file.
        :type filename: str
        :return: DataFrame with project data.
        :rtype: pd.DataFrame
        :raises FileNotFoundError: If processed data not found.

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_jira_data(filename="jira_projects.csv")
        """
        if processed_dir is None:
            processed_dir = Path(__file__).parent.parent.parent / "data" / "processed"

        file_path = Path(processed_dir) / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed JIRA data not found: {file_path}\n"
                f"Please run the preprocessing script first:\n"
                f"  python scripts/preprocess_jira_data.py --sample 50"
            )

        return self.load(file_path)

    def load_sample_data(self) -> pd.DataFrame:
        """
        Load the sample projects data for testing/demo.

        :return: DataFrame with sample project data.
        :rtype: pd.DataFrame
        :raises FileNotFoundError: If sample data not found.

        Example:
            >>> loader = DataLoader()
            >>> sample_df = loader.load_sample_data()
        """
        sample_path = (
            Path(__file__).parent.parent.parent / "data" / "raw" / "sample_projects.csv"
        )

        if not sample_path.exists():
            raise FileNotFoundError(f"Sample data not found: {sample_path}")

        return self.load(sample_path)

    def load_from_bytes(
        self,
        file_bytes: bytes,
        file_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load data from bytes (e.g., from Streamlit file uploader).

        :param file_bytes: Raw file bytes.
        :type file_bytes: bytes
        :param file_name: Original filename (used to determine format).
        :type file_name: str
        :param kwargs: Additional arguments for the loader.
        :return: DataFrame containing the loaded data.
        :rtype: pd.DataFrame
        :raises ValueError: If format is not supported.

        Example:
            >>> loader = DataLoader()
            >>> df = loader.load_from_bytes(uploaded_file.read(), "data.csv")
        """
        from io import BytesIO, StringIO

        suffix = Path(file_name).suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(StringIO(file_bytes.decode("utf-8")), **kwargs)
        if suffix == ".json":
            data = json.loads(file_bytes.decode("utf-8"))
            if isinstance(data, list):
                return pd.DataFrame(data)
            if isinstance(data, dict) and "projects" in data:
                return pd.DataFrame(data["projects"])
            return pd.DataFrame([data])
        if suffix in [".xlsx", ".xls"]:
            return pd.read_excel(BytesIO(file_bytes), **kwargs)

        raise ValueError(f"Unsupported format: {suffix}")
