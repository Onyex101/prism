"""
Data Preprocessor Module

Handles data cleaning, transformation, and preparation for ML models.
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """Preprocess project data for machine learning."""

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scaling_method: str = "standard",
    ):
        """
        Initialize the preprocessor.

        Args:
            numerical_strategy: How to handle missing numerical values
            categorical_strategy: How to handle missing categorical values
            scaling_method: Scaling method for numerical features
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.fitted = False

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Fit preprocessors and transform data.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df, numerical_cols, categorical_cols)

        # Encode categorical variables
        df = self._encode_categoricals(df, categorical_cols, fit=True)

        # Scale numerical features
        df = self._scale_numericals(df, numerical_cols, fit=True)

        self.fitted = True
        logger.info("Preprocessing complete (fit_transform)")

        return df

    def transform(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessors.

        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
            categorical_cols: List of categorical column names

        Returns:
            Preprocessed DataFrame
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()

        df = self._handle_missing_values(df, numerical_cols, categorical_cols)
        df = self._encode_categoricals(df, categorical_cols, fit=False)
        df = self._scale_numericals(df, numerical_cols, fit=False)

        logger.info("Preprocessing complete (transform)")

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """Handle missing values in the data."""
        # Numerical columns
        for col in numerical_cols:
            if col in df.columns and df[col].isna().any():
                if self.numerical_strategy == "median":
                    fill_value = df[col].median()
                elif self.numerical_strategy == "mean":
                    fill_value = df[col].mean()
                elif self.numerical_strategy == "zero":
                    fill_value = 0
                else:
                    fill_value = df[col].median()

                df[col] = df[col].fillna(fill_value)
                logger.debug(f"Filled {col} missing values with {fill_value}")

        # Categorical columns
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                if self.categorical_strategy == "most_frequent":
                    fill_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "Unknown"
                else:
                    fill_value = "Unknown"

                df[col] = df[col].fillna(fill_value)

        # Text columns - fill with empty string
        text_cols = ["status_comments", "project_description", "team_feedback"]
        for col in text_cols:
            if col in df.columns:
                df[col] = df[col].fillna("")

        return df

    def _encode_categoricals(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Encode categorical variables."""
        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                # Handle unseen categories by adding 'Unknown'
                unique_values = df[col].astype(str).unique().tolist()
                if "Unknown" not in unique_values:
                    unique_values.append("Unknown")
                le.fit(unique_values)
                self.label_encoders[col] = le

            if col in self.label_encoders:
                le = self.label_encoders[col]
                # Transform with handling for unseen values
                df[f"{col}_encoded"] = df[col].astype(str).apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else le.transform(["Unknown"])[0]
                )

        return df

    def _scale_numericals(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """Scale numerical features."""
        cols_to_scale = [col for col in numerical_cols if col in df.columns]

        if not cols_to_scale:
            return df

        if fit:
            self.scaler = StandardScaler()
            df[cols_to_scale] = self.scaler.fit_transform(df[cols_to_scale])
        elif self.scaler is not None:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])

        return df

    def clean_text(self, text: str) -> str:
        """Clean text data for LLM analysis."""
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace

        return text

    def prepare_for_ml(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare data for ML model training/prediction.

        Args:
            df: Preprocessed DataFrame
            target_col: Name of target column (if training)

        Returns:
            Tuple of (features DataFrame, target Series or None)
        """
        # Drop non-feature columns
        drop_cols = [
            "project_id",
            "project_name",
            "status_comments",
            "project_description",
            "team_feedback",
            "stakeholder_notes",
            "start_date",
            "planned_end_date",
            "actual_end_date",
            "technology_stack",
        ]

        if target_col:
            drop_cols.append(target_col)
            y = df[target_col] if target_col in df.columns else None
        else:
            y = None

        # Keep only feature columns
        feature_cols = [col for col in df.columns if col not in drop_cols]
        X = df[feature_cols]

        return X, y
