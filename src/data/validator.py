"""
Data Validator Module
=====================

This module provides data validation functionality for project data.

It validates data against schema definitions and business rules to ensure
data quality before processing.

Example:
    >>> from src.data.validator import DataValidator
    >>> validator = DataValidator()
    >>> result = validator.validate(df)
    >>> if result.is_valid:
    ...     print("Data is valid!")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from loguru import logger


@dataclass
class ValidationResult:
    """
    Result of data validation.

    This dataclass encapsulates the results of a validation operation,
    including errors, warnings, and statistics about the data.

    :ivar is_valid: Whether the data passed validation.
    :vartype is_valid: bool
    :ivar errors: List of validation errors found.
    :vartype errors: list[dict[str, Any]]
    :ivar warnings: List of validation warnings found.
    :vartype warnings: list[dict[str, Any]]
    :ivar stats: Statistics about the validated data.
    :vartype stats: dict[str, Any]

    Example:
        >>> result = ValidationResult(is_valid=True)
        >>> print(result.error_count)
        0
    """

    is_valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        """
        Get the number of errors.

        :return: Count of validation errors.
        :rtype: int
        """
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        """
        Get the number of warnings.

        :return: Count of validation warnings.
        :rtype: int
        """
        return len(self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to dictionary.

        :return: Dictionary representation of the result.
        :rtype: dict[str, Any]
        """
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class DataValidator:
    """
    Validate project data against schema and rules.

    This class follows the Single Responsibility Principle by focusing
    solely on data validation. Validation rules are modular and can be
    extended by subclassing.

    :cvar REQUIRED_COLUMNS: List of required column names.
    :vartype REQUIRED_COLUMNS: list[str]
    :cvar VALID_STATUSES: Valid status values.
    :vartype VALID_STATUSES: list[str]
    :cvar VALID_PRIORITIES: Valid priority values.
    :vartype VALID_PRIORITIES: list[str]

    :ivar rules_path: Path to validation rules YAML file.
    :vartype rules_path: Optional[Path]
    :ivar rules: Loaded validation rules.
    :vartype rules: dict

    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate(projects_df)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(error["message"])
    """

    REQUIRED_COLUMNS: list[str] = [
        "project_id",
        "project_name",
        "start_date",
        "planned_end_date",
        "budget",
        "spent",
        "planned_hours",
        "actual_hours",
        "team_size",
        "completion_rate",
        "status",
        "status_comments",
    ]

    VALID_STATUSES: list[str] = [
        "Active",
        "On Hold",
        "Completed",
        "Cancelled",
        "Planning",
    ]

    VALID_PRIORITIES: list[str] = ["Critical", "High", "Medium", "Low"]

    def __init__(self, rules_path: Optional[Path] = None) -> None:
        """
        Initialize the validator.

        :param rules_path: Path to validation rules YAML file.
        :type rules_path: Optional[Path]
        """
        self.rules_path = rules_path
        self.rules = self._load_rules() if rules_path else {}

    def _load_rules(self) -> dict:
        """
        Load validation rules from YAML file.

        :return: Dictionary of validation rules.
        :rtype: dict
        """
        if self.rules_path and self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against all rules.

        This method runs all validation checks and returns a comprehensive
        result object containing errors, warnings, and statistics.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: Validation result with errors, warnings, and stats.
        :rtype: ValidationResult

        Example:
            >>> result = validator.validate(df)
            >>> print(f"Valid: {result.is_valid}, Errors: {result.error_count}")
        """
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        # Run all validation checks
        errors.extend(self._validate_required_columns(df))
        errors.extend(self._validate_unique_ids(df))

        field_errors, field_warnings = self._validate_fields(df)
        errors.extend(field_errors)
        warnings.extend(field_warnings)

        cross_errors, cross_warnings = self._validate_cross_fields(df)
        errors.extend(cross_errors)
        warnings.extend(cross_warnings)

        stats = self._calculate_stats(df)
        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

        self._log_result(result)
        return result

    def _validate_required_columns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Check for missing required columns.

        :param df: DataFrame to check.
        :type df: pd.DataFrame
        :return: List of error dictionaries.
        :rtype: list[dict[str, Any]]
        """
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            return [
                {
                    "type": "missing_required_columns",
                    "message": f"Missing required columns: {missing_cols}",
                    "columns": missing_cols,
                }
            ]
        return []

    def _validate_unique_ids(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Check for duplicate project IDs.

        :param df: DataFrame to check.
        :type df: pd.DataFrame
        :return: List of error dictionaries.
        :rtype: list[dict[str, Any]]
        """
        if "project_id" not in df.columns:
            return []

        duplicates = df[df["project_id"].duplicated()]["project_id"].tolist()
        if duplicates:
            return [
                {
                    "type": "duplicate_project_id",
                    "message": f"Duplicate project IDs found: {duplicates}",
                    "duplicates": duplicates,
                }
            ]
        return []

    def _validate_fields(
        self, df: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate individual field values.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: Tuple of (errors, warnings).
        :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
        """
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        errors.extend(self._validate_numeric_fields(df))
        errors.extend(self._validate_completion_rate(df))
        warnings.extend(self._validate_status_values(df))
        warnings.extend(self._validate_text_lengths(df))

        return errors, warnings

    def _validate_numeric_fields(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate that numeric fields are positive.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: List of error dictionaries.
        :rtype: list[dict[str, Any]]
        """
        errors: list[dict[str, Any]] = []
        numeric_fields = ["budget", "spent", "planned_hours", "actual_hours", "team_size"]

        for field_name in numeric_fields:
            if field_name in df.columns:
                negative_count = (df[field_name] < 0).sum()
                if negative_count > 0:
                    errors.append(
                        {
                            "type": "negative_values",
                            "field": field_name,
                            "message": f"{field_name} has {negative_count} negative values",
                            "count": int(negative_count),
                        }
                    )
        return errors

    def _validate_completion_rate(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate that completion_rate is between 0 and 100.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: List of error dictionaries.
        :rtype: list[dict[str, Any]]
        """
        if "completion_rate" not in df.columns:
            return []

        invalid = ((df["completion_rate"] < 0) | (df["completion_rate"] > 100)).sum()
        if invalid > 0:
            return [
                {
                    "type": "invalid_completion_rate",
                    "message": f"completion_rate must be 0-100, found {invalid} invalid values",
                    "count": int(invalid),
                }
            ]
        return []

    def _validate_status_values(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate status values against allowed values.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: List of warning dictionaries.
        :rtype: list[dict[str, Any]]
        """
        if "status" not in df.columns:
            return []

        invalid_statuses = df[~df["status"].isin(self.VALID_STATUSES)]["status"].unique()
        if len(invalid_statuses) > 0:
            return [
                {
                    "type": "invalid_status",
                    "message": f"Unknown status values: {list(invalid_statuses)}",
                    "values": list(invalid_statuses),
                }
            ]
        return []

    def _validate_text_lengths(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Validate text field lengths meet minimum requirements.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: List of warning dictionaries.
        :rtype: list[dict[str, Any]]
        """
        if "status_comments" not in df.columns:
            return []

        short_comments = (df["status_comments"].str.len() < 100).sum()
        if short_comments > 0:
            return [
                {
                    "type": "short_text",
                    "field": "status_comments",
                    "message": f"{short_comments} projects have comments < 100 characters",
                    "count": int(short_comments),
                }
            ]
        return []

    def _validate_cross_fields(
        self, df: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate relationships between fields.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: Tuple of (errors, warnings).
        :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
        """
        errors: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []

        date_errors, date_warnings = self._validate_date_ranges(df)
        errors.extend(date_errors)
        warnings.extend(date_warnings)

        warnings.extend(self._validate_budget_overruns(df))

        return errors, warnings

    def _validate_date_ranges(
        self, df: pd.DataFrame
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Validate that end dates are after start dates.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: Tuple of (errors, warnings).
        :rtype: tuple[list[dict[str, Any]], list[dict[str, Any]]]
        """
        if "start_date" not in df.columns or "planned_end_date" not in df.columns:
            return [], []

        try:
            start = pd.to_datetime(df["start_date"])
            end = pd.to_datetime(df["planned_end_date"])
            invalid = (end < start).sum()
            if invalid > 0:
                return [
                    {
                        "type": "invalid_date_range",
                        "message": f"{invalid} projects have end date before start date",
                        "count": int(invalid),
                    }
                ], []
        except Exception as e:
            return [], [
                {
                    "type": "date_parse_error",
                    "message": f"Could not parse dates: {str(e)}",
                }
            ]
        return [], []

    def _validate_budget_overruns(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Check for extreme budget overruns.

        :param df: DataFrame to validate.
        :type df: pd.DataFrame
        :return: List of warning dictionaries.
        :rtype: list[dict[str, Any]]
        """
        if "spent" not in df.columns or "budget" not in df.columns:
            return []

        mask = (df["budget"] > 0) & (df["spent"] > df["budget"] * 2)
        over_budget = mask.sum()
        if over_budget > 0:
            return [
                {
                    "type": "extreme_budget_overrun",
                    "message": f"{over_budget} projects have spent > 2x budget",
                    "count": int(over_budget),
                }
            ]
        return []

    def _calculate_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Calculate data quality statistics.

        :param df: DataFrame to analyze.
        :type df: pd.DataFrame
        :return: Dictionary of statistics.
        :rtype: dict[str, Any]
        """
        total_rows = len(df)
        total_cols = len(df.columns)

        completeness = self._calculate_completeness(df, total_rows)
        overall_completeness = (
            sum(completeness.values()) / len(completeness) if completeness else 0
        )

        return {
            "total_projects": total_rows,
            "total_columns": total_cols,
            "overall_completeness_pct": round(overall_completeness, 1),
            "column_completeness": completeness,
            "required_columns_present": len(
                [c for c in self.REQUIRED_COLUMNS if c in df.columns]
            ),
            "required_columns_total": len(self.REQUIRED_COLUMNS),
        }

    def _calculate_completeness(
        self, df: pd.DataFrame, total_rows: int
    ) -> dict[str, float]:
        """
        Calculate completeness percentage for each column.

        :param df: DataFrame to analyze.
        :type df: pd.DataFrame
        :param total_rows: Total number of rows.
        :type total_rows: int
        :return: Dictionary mapping column names to completeness percentages.
        :rtype: dict[str, float]
        """
        completeness: dict[str, float] = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = (
                round(non_null / total_rows * 100, 1) if total_rows > 0 else 0
            )
        return completeness

    def _log_result(self, result: ValidationResult) -> None:
        """
        Log the validation result.

        :param result: Validation result to log.
        :type result: ValidationResult
        """
        if result.is_valid:
            logger.info(f"Validation passed with {result.warning_count} warnings")
        else:
            logger.warning(f"Validation failed with {result.error_count} errors")
