"""
Data Module
===========

This module provides data loading, validation, preprocessing, feature
engineering, and synthetic data generation functionality.

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

Example:
    >>> from src.data import DataLoader, DataValidator, FeatureEngineer
    >>> loader = DataLoader()
    >>> df = loader.load("data/raw/projects.csv")
    >>> validator = DataValidator()
    >>> result = validator.validate(df)
"""

from src.data.feature_engineer import FeatureEngineer
from src.data.generator import SyntheticDataGenerator
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.validator import DataValidator, ValidationResult

__all__ = [
    "DataLoader",
    "DataValidator",
    "ValidationResult",
    "DataPreprocessor",
    "FeatureEngineer",
    "SyntheticDataGenerator",
]
