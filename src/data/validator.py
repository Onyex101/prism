"""
Data Validator Module

Validates project data against schema and business rules.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from loguru import logger


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


class DataValidator:
    """Validate project data against schema and rules."""

    REQUIRED_COLUMNS = [
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

    VALID_STATUSES = ["Active", "On Hold", "Completed", "Cancelled", "Planning"]
    VALID_PRIORITIES = ["Critical", "High", "Medium", "Low"]

    def __init__(self, rules_path: Optional[Path] = None):
        """
        Initialize the validator.

        Args:
            rules_path: Path to validation rules YAML file
        """
        self.rules_path = rules_path
        self.rules = self._load_rules() if rules_path else {}

    def _load_rules(self) -> dict:
        """Load validation rules from YAML file."""
        if self.rules_path and self.rules_path.exists():
            with open(self.rules_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against all rules.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with errors, warnings, and stats
        """
        errors = []
        warnings = []

        # Check required columns
        missing_cols = self._check_required_columns(df)
        if missing_cols:
            errors.append({
                "type": "missing_required_columns",
                "message": f"Missing required columns: {missing_cols}",
                "columns": missing_cols,
            })

        # Check for duplicate project IDs
        if "project_id" in df.columns:
            duplicates = self._check_duplicates(df, "project_id")
            if duplicates:
                errors.append({
                    "type": "duplicate_project_id",
                    "message": f"Duplicate project IDs found: {duplicates}",
                    "duplicates": duplicates,
                })

        # Validate individual fields
        field_errors, field_warnings = self._validate_fields(df)
        errors.extend(field_errors)
        warnings.extend(field_warnings)

        # Cross-field validation
        cross_errors, cross_warnings = self._validate_cross_fields(df)
        errors.extend(cross_errors)
        warnings.extend(cross_warnings)

        # Calculate stats
        stats = self._calculate_stats(df)

        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats,
        )

        if is_valid:
            logger.info(f"Validation passed with {len(warnings)} warnings")
        else:
            logger.warning(f"Validation failed with {len(errors)} errors")

        return result

    def _check_required_columns(self, df: pd.DataFrame) -> list[str]:
        """Check for missing required columns."""
        return [col for col in self.REQUIRED_COLUMNS if col not in df.columns]

    def _check_duplicates(self, df: pd.DataFrame, column: str) -> list:
        """Check for duplicate values in a column."""
        duplicated = df[df[column].duplicated()][column].tolist()
        return duplicated

    def _validate_fields(self, df: pd.DataFrame) -> tuple[list, list]:
        """Validate individual field values."""
        errors = []
        warnings = []

        # Validate numeric fields are positive
        numeric_fields = ["budget", "spent", "planned_hours", "actual_hours", "team_size"]
        for field in numeric_fields:
            if field in df.columns:
                negative_count = (df[field] < 0).sum()
                if negative_count > 0:
                    errors.append({
                        "type": "negative_values",
                        "field": field,
                        "message": f"{field} has {negative_count} negative values",
                        "count": int(negative_count),
                    })

        # Validate completion_rate is 0-100
        if "completion_rate" in df.columns:
            invalid = ((df["completion_rate"] < 0) | (df["completion_rate"] > 100)).sum()
            if invalid > 0:
                errors.append({
                    "type": "invalid_completion_rate",
                    "message": f"completion_rate must be 0-100, found {invalid} invalid values",
                    "count": int(invalid),
                })

        # Validate status values
        if "status" in df.columns:
            invalid_statuses = df[~df["status"].isin(self.VALID_STATUSES)]["status"].unique()
            if len(invalid_statuses) > 0:
                warnings.append({
                    "type": "invalid_status",
                    "message": f"Unknown status values: {list(invalid_statuses)}",
                    "values": list(invalid_statuses),
                })

        # Check text field lengths
        if "status_comments" in df.columns:
            short_comments = (df["status_comments"].str.len() < 100).sum()
            if short_comments > 0:
                warnings.append({
                    "type": "short_text",
                    "field": "status_comments",
                    "message": f"{short_comments} projects have comments < 100 characters",
                    "count": int(short_comments),
                })

        return errors, warnings

    def _validate_cross_fields(self, df: pd.DataFrame) -> tuple[list, list]:
        """Validate relationships between fields."""
        errors = []
        warnings = []

        # Check end date after start date
        if "start_date" in df.columns and "planned_end_date" in df.columns:
            try:
                start = pd.to_datetime(df["start_date"])
                end = pd.to_datetime(df["planned_end_date"])
                invalid = (end < start).sum()
                if invalid > 0:
                    errors.append({
                        "type": "invalid_date_range",
                        "message": f"{invalid} projects have end date before start date",
                        "count": int(invalid),
                    })
            except Exception as e:
                warnings.append({
                    "type": "date_parse_error",
                    "message": f"Could not parse dates: {str(e)}",
                })

        # Check spent vs budget (warning if >2x)
        if "spent" in df.columns and "budget" in df.columns:
            mask = (df["budget"] > 0) & (df["spent"] > df["budget"] * 2)
            over_budget = mask.sum()
            if over_budget > 0:
                warnings.append({
                    "type": "extreme_budget_overrun",
                    "message": f"{over_budget} projects have spent > 2x budget",
                    "count": int(over_budget),
                })

        return errors, warnings

    def _calculate_stats(self, df: pd.DataFrame) -> dict:
        """Calculate data quality statistics."""
        total_rows = len(df)
        total_cols = len(df.columns)

        # Calculate completeness
        completeness = {}
        for col in df.columns:
            non_null = df[col].notna().sum()
            completeness[col] = round(non_null / total_rows * 100, 1) if total_rows > 0 else 0

        overall_completeness = sum(completeness.values()) / len(completeness) if completeness else 0

        return {
            "total_projects": total_rows,
            "total_columns": total_cols,
            "overall_completeness_pct": round(overall_completeness, 1),
            "column_completeness": completeness,
            "required_columns_present": len([c for c in self.REQUIRED_COLUMNS if c in df.columns]),
            "required_columns_total": len(self.REQUIRED_COLUMNS),
        }
