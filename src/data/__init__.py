"""
Data Module
===========

This module provides data loading, validation, preprocessing, feature
engineering, and synthetic data generation functionality.

Heavy dependencies (e.g. scikit-learn) are loaded lazily via :func:`__getattr__`
so ``import src.data.jira_aggregator`` does not import sklearn until needed.

Data Locations:
    - ``data/raw/``: Real project data files (CSV, Excel, JSON)
    - ``data/processed/``: Preprocessed and feature-engineered data
    - ``data/schemas/``: Validation schemas and rules

Classes:
    DataLoader: Load project data from various file formats.
    DataValidator: Validate project data against schema and rules.
    ValidationResult: Result container for validation operations.
    DataPreprocessor: Preprocess data for ML models.
    FeatureEngineer: Create derived features from raw data.
    SyntheticDataGenerator: Generate synthetic data for testing/demos.
    JiraAggregator: Aggregate Jira issues to PRISM project rows.

Example:
    >>> from src.data import DataLoader, DataValidator, FeatureEngineer
    >>> loader = DataLoader()
    >>> df = loader.load("data/raw/projects.csv")
    >>> validator = DataValidator()
    >>> result = validator.validate(df)
"""

from __future__ import annotations

from typing import Any

from src.data.loader import DataLoader
from src.data.validator import DataValidator, ValidationResult

__all__ = [
    "DataLoader",
    "DataValidator",
    "ValidationResult",
    "DataPreprocessor",
    "FeatureEngineer",
    "SyntheticDataGenerator",
    "JiraAggregator",
]


def __getattr__(name: str) -> Any:
    """Lazy-import optional / heavy submodules (sklearn, etc.)."""
    if name == "DataPreprocessor":
        from src.data.preprocessor import DataPreprocessor

        return DataPreprocessor
    if name == "FeatureEngineer":
        from src.data.feature_engineer import FeatureEngineer

        return FeatureEngineer
    if name == "SyntheticDataGenerator":
        from src.data.generator import SyntheticDataGenerator

        return SyntheticDataGenerator
    if name == "JiraAggregator":
        from src.data.jira_aggregator import JiraAggregator

        return JiraAggregator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
