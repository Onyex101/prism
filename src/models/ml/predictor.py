"""
ML Predictor Module
===================

This module handles loading trained models and making predictions.

It provides a simple interface for loading serialized models and
generating risk predictions and scores.

Example:
    >>> from src.models.ml.predictor import MLPredictor
    >>> predictor = MLPredictor(model_path="models/ml/best_model.pkl")
    >>> predictions = predictor.predict(X_new)
"""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class MLPredictor:
    """
    Load and use trained ML models for risk prediction.

    This class follows the Single Responsibility Principle by focusing
    solely on prediction operations. It handles model loading and provides
    multiple prediction methods.

    :ivar model: Loaded model instance.
    :vartype model: Optional[Any]
    :ivar scaler: Loaded scaler instance.
    :vartype scaler: Optional[Any]
    :ivar feature_names: Expected feature names.
    :vartype feature_names: list[str]

    Example:
        >>> predictor = MLPredictor("models/ml/model.pkl")
        >>> risk_scores = predictor.get_risk_scores(X_test)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize the predictor.

        :param model_path: Path to trained model file.
        :type model_path: Optional[Union[str, Path]]
        :param scaler_path: Path to fitted scaler file.
        :type scaler_path: Optional[Union[str, Path]]
        """
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []

        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.

        :param model_path: Path to model file.
        :type model_path: Union[str, Path]
        :raises FileNotFoundError: If model file not found.

        Example:
            >>> predictor = MLPredictor()
            >>> predictor.load_model("models/ml/random_forest.pkl")
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        self._load_feature_names(model_path)

    def _load_feature_names(self, model_path: Path) -> None:
        """
        Load feature names from associated JSON file.

        :param model_path: Path to model file.
        :type model_path: Path
        """
        feature_path = model_path.parent / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, "r") as f:
                self.feature_names = json.load(f)

    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """
        Load a fitted scaler from disk.

        :param scaler_path: Path to scaler file.
        :type scaler_path: Union[str, Path]

        Example:
            >>> predictor.load_scaler("models/scalers/standard_scaler.pkl")
        """
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk categories.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: Array of predicted risk categories.
        :rtype: np.ndarray
        :raises ValueError: If no model loaded.

        Example:
            >>> predictions = predictor.predict(X_test)
            >>> print(predictions[:5])
        """
        self._validate_model_loaded()
        X_processed = self._apply_scaler(X)
        predictions = self.model.predict(X_processed)
        logger.debug(f"Made predictions for {len(predictions)} samples")
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk probabilities.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: Array of probability scores (probability of high risk).
        :rtype: np.ndarray
        :raises ValueError: If no model loaded.

        Example:
            >>> probas = predictor.predict_proba(X_test)
            >>> high_risk_mask = probas >= 0.6
        """
        self._validate_model_loaded()
        X_processed = self._apply_scaler(X)

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_processed)
            return probas[:, -1] if probas.ndim > 1 else probas

        # Fallback for models without predict_proba
        return self.model.predict(X_processed)

    def get_risk_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get risk scores with metadata.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: DataFrame with risk_score and risk_level columns.
        :rtype: pd.DataFrame

        Example:
            >>> scores_df = predictor.get_risk_scores(X_test)
            >>> high_risk = scores_df[scores_df["risk_level"] == "High"]
        """
        probas = self.predict_proba(X)
        risk_levels = [self._classify_risk_level(p) for p in probas]

        return pd.DataFrame(
            {
                "risk_score": probas,
                "risk_level": risk_levels,
            }
        )

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model.

        :return: DataFrame with feature names and importance scores,
            or None if not available.
        :rtype: Optional[pd.DataFrame]

        Example:
            >>> importance_df = predictor.get_feature_importance()
            >>> top_features = importance_df.head(10)
        """
        if self.model is None:
            return None

        importance = self._extract_importance()
        if importance is None:
            return None

        feature_names = self._get_feature_names(len(importance))

        return pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

    def _validate_model_loaded(self) -> None:
        """
        Validate that a model is loaded.

        :raises ValueError: If no model loaded.
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

    def _apply_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply scaler to data if available.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: Scaled data.
        :rtype: np.ndarray
        """
        if self.scaler is not None:
            return self.scaler.transform(X)
        return X.values if isinstance(X, pd.DataFrame) else X

    def _classify_risk_level(self, probability: float) -> str:
        """
        Classify risk level based on probability.

        :param probability: Risk probability.
        :type probability: float
        :return: Risk level string.
        :rtype: str
        """
        if probability >= 0.6:
            return "High"
        if probability >= 0.3:
            return "Medium"
        return "Low"

    def _extract_importance(self) -> Optional[np.ndarray]:
        """
        Extract feature importance from model.

        :return: Array of importance values or None.
        :rtype: Optional[np.ndarray]
        """
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        if hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).flatten()
        return None

    def _get_feature_names(self, n_features: int) -> list[str]:
        """
        Get feature names, generating defaults if needed.

        :param n_features: Number of features.
        :type n_features: int
        :return: List of feature names.
        :rtype: list[str]
        """
        if self.feature_names:
            return self.feature_names[:n_features]
        return [f"feature_{i}" for i in range(n_features)]
