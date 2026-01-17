"""
Data Preprocessor Module
========================

This module handles data cleaning, transformation, and preparation for ML models.

It provides functionality for handling missing values, encoding categorical
variables, and scaling numerical features.

Example:
    >>> from src.data.preprocessor import DataPreprocessor
    >>> preprocessor = DataPreprocessor()
    >>> df_processed = preprocessor.fit_transform(df, numerical_cols, categorical_cols)
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataPreprocessor:
    """
    Preprocess project data for machine learning.

    This class follows the Single Responsibility Principle by focusing
    solely on data preprocessing operations. It provides a scikit-learn
    compatible interface with fit_transform and transform methods.

    :ivar numerical_strategy: Strategy for handling missing numerical values.
        Options: "median", "mean", "zero".
    :vartype numerical_strategy: str
    :ivar categorical_strategy: Strategy for handling missing categorical values.
        Options: "most_frequent", "unknown".
    :vartype categorical_strategy: str
    :ivar scaling_method: Method for scaling numerical features.
    :vartype scaling_method: str
    :ivar scaler: Fitted StandardScaler instance.
    :vartype scaler: Optional[StandardScaler]
    :ivar label_encoders: Dictionary of fitted LabelEncoders.
    :vartype label_encoders: dict[str, LabelEncoder]
    :ivar fitted: Whether the preprocessor has been fitted.
    :vartype fitted: bool

    Example:
        >>> preprocessor = DataPreprocessor(numerical_strategy="median")
        >>> df_train = preprocessor.fit_transform(df, num_cols, cat_cols)
        >>> df_test = preprocessor.transform(df_test, num_cols, cat_cols)
    """

    def __init__(
        self,
        numerical_strategy: str = "median",
        categorical_strategy: str = "most_frequent",
        scaling_method: str = "standard",
    ) -> None:
        """
        Initialize the preprocessor.

        :param numerical_strategy: How to handle missing numerical values.
            Options: "median" (default), "mean", "zero".
        :type numerical_strategy: str
        :param categorical_strategy: How to handle missing categorical values.
            Options: "most_frequent" (default), "unknown".
        :type categorical_strategy: str
        :param scaling_method: Scaling method for numerical features.
            Options: "standard" (default).
        :type scaling_method: str
        """
        self.numerical_strategy = numerical_strategy
        self.categorical_strategy = categorical_strategy
        self.scaling_method = scaling_method

        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.fitted = False
        self._numerical_fill_values: dict[str, float] = {}

    def fit_transform(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Fit preprocessors and transform data.

        This method fits the scaler and encoders on the data and then
        transforms it. Use this for training data.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :param numerical_cols: List of numerical column names.
        :type numerical_cols: list[str]
        :param categorical_cols: List of categorical column names.
        :type categorical_cols: list[str]
        :return: Preprocessed DataFrame.
        :rtype: pd.DataFrame

        Example:
            >>> df_processed = preprocessor.fit_transform(
            ...     df,
            ...     ["budget", "spent", "team_size"],
            ...     ["status", "priority"]
            ... )
        """
        df = df.copy()

        df = self._handle_missing_values(df, numerical_cols, categorical_cols, fit=True)
        df = self._encode_categoricals(df, categorical_cols, fit=True)
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

        This method applies the previously fitted transformations.
        Use this for validation/test data.

        :param df: Input DataFrame.
        :type df: pd.DataFrame
        :param numerical_cols: List of numerical column names.
        :type numerical_cols: list[str]
        :param categorical_cols: List of categorical column names.
        :type categorical_cols: list[str]
        :return: Preprocessed DataFrame.
        :rtype: pd.DataFrame
        :raises ValueError: If preprocessor not fitted.

        Example:
            >>> df_test_processed = preprocessor.transform(
            ...     df_test, num_cols, cat_cols
            ... )
        """
        if not self.fitted:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")

        df = df.copy()

        df = self._handle_missing_values(
            df, numerical_cols, categorical_cols, fit=False
        )
        df = self._encode_categoricals(df, categorical_cols, fit=False)
        df = self._scale_numericals(df, numerical_cols, fit=False)

        logger.info("Preprocessing complete (transform)")

        return df

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        categorical_cols: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :param numerical_cols: Numerical column names.
        :type numerical_cols: list[str]
        :param categorical_cols: Categorical column names.
        :type categorical_cols: list[str]
        :param fit: Whether to fit fill values or use existing.
        :type fit: bool
        :return: DataFrame with missing values handled.
        :rtype: pd.DataFrame
        """
        df = self._handle_numerical_missing(df, numerical_cols, fit)
        df = self._handle_categorical_missing(df, categorical_cols)
        df = self._handle_text_missing(df)
        return df

    def _handle_numerical_missing(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        fit: bool,
    ) -> pd.DataFrame:
        """
        Handle missing values in numerical columns.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :param numerical_cols: Numerical column names.
        :type numerical_cols: list[str]
        :param fit: Whether to compute fill values.
        :type fit: bool
        :return: DataFrame with numerical missing values handled.
        :rtype: pd.DataFrame
        """
        for col in numerical_cols:
            if col not in df.columns:
                continue

            if df[col].isna().any():
                if fit:
                    fill_value = self._compute_fill_value(df[col])
                    self._numerical_fill_values[col] = fill_value
                else:
                    fill_value = self._numerical_fill_values.get(col, 0)

                df[col] = df[col].fillna(fill_value)
                logger.debug(f"Filled {col} missing values with {fill_value}")

        return df

    def _compute_fill_value(self, series: pd.Series) -> float:
        """
        Compute fill value based on strategy.

        :param series: Series to compute fill value for.
        :type series: pd.Series
        :return: Fill value.
        :rtype: float
        """
        if self.numerical_strategy == "median":
            return series.median()
        if self.numerical_strategy == "mean":
            return series.mean()
        if self.numerical_strategy == "zero":
            return 0
        return series.median()

    def _handle_categorical_missing(
        self,
        df: pd.DataFrame,
        categorical_cols: list[str],
    ) -> pd.DataFrame:
        """
        Handle missing values in categorical columns.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :param categorical_cols: Categorical column names.
        :type categorical_cols: list[str]
        :return: DataFrame with categorical missing values handled.
        :rtype: pd.DataFrame
        """
        for col in categorical_cols:
            if col not in df.columns or not df[col].isna().any():
                continue

            if self.categorical_strategy == "most_frequent":
                mode_values = df[col].mode()
                fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
            else:
                fill_value = "Unknown"

            df[col] = df[col].fillna(fill_value)

        return df

    def _handle_text_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in text columns.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :return: DataFrame with text missing values handled.
        :rtype: pd.DataFrame
        """
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
        """
        Encode categorical variables.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :param categorical_cols: Categorical column names.
        :type categorical_cols: list[str]
        :param fit: Whether to fit encoders.
        :type fit: bool
        :return: DataFrame with encoded categoricals.
        :rtype: pd.DataFrame
        """
        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                le = LabelEncoder()
                unique_values = df[col].astype(str).unique().tolist()
                if "Unknown" not in unique_values:
                    unique_values.append("Unknown")
                le.fit(unique_values)
                self.label_encoders[col] = le

            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[f"{col}_encoded"] = df[col].astype(str).apply(
                    lambda x: (
                        le.transform([x])[0]
                        if x in le.classes_
                        else le.transform(["Unknown"])[0]
                    )
                )

        return df

    def _scale_numericals(
        self,
        df: pd.DataFrame,
        numerical_cols: list[str],
        fit: bool = True,
    ) -> pd.DataFrame:
        """
        Scale numerical features.

        :param df: DataFrame to process.
        :type df: pd.DataFrame
        :param numerical_cols: Numerical column names.
        :type numerical_cols: list[str]
        :param fit: Whether to fit scaler.
        :type fit: bool
        :return: DataFrame with scaled numericals.
        :rtype: pd.DataFrame
        """
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
        """
        Clean text data for LLM analysis.

        :param text: Text to clean.
        :type text: str
        :return: Cleaned text.
        :rtype: str

        Example:
            >>> cleaned = preprocessor.clean_text("  Multiple   spaces  ")
            "Multiple spaces"
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

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

        This method separates features from target and drops non-feature
        columns like IDs and text fields.

        :param df: Preprocessed DataFrame.
        :type df: pd.DataFrame
        :param target_col: Name of target column (if training).
        :type target_col: Optional[str]
        :return: Tuple of (features DataFrame, target Series or None).
        :rtype: Tuple[pd.DataFrame, Optional[pd.Series]]

        Example:
            >>> X, y = preprocessor.prepare_for_ml(df, target_col="risk_level")
        """
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

        y: Optional[pd.Series] = None
        if target_col:
            drop_cols.append(target_col)
            y = df[target_col] if target_col in df.columns else None

        feature_cols = [col for col in df.columns if col not in drop_cols]
        X = df[feature_cols]

        return X, y
