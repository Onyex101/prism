"""
ML Predictor Module

Handles loading trained models and making predictions.
"""

import json
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger


class MLPredictor:
    """Load and use trained ML models for risk prediction."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model file
            scaler_path: Path to fitted scaler file
        """
        self.model = None
        self.scaler = None
        self.feature_names: list[str] = []

        if model_path:
            self.load_model(model_path)
        if scaler_path:
            self.load_scaler(scaler_path)

    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model from disk."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")

        # Try to load feature names
        feature_path = model_path.parent / "feature_names.json"
        if feature_path.exists():
            with open(feature_path, "r") as f:
                self.feature_names = json.load(f)

    def load_scaler(self, scaler_path: Union[str, Path]) -> None:
        """Load a fitted scaler from disk."""
        scaler_path = Path(scaler_path)
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk categories.

        Args:
            X: Feature matrix

        Returns:
            Array of predicted risk categories
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X

        predictions = self.model.predict(X_scaled)
        logger.debug(f"Made predictions for {len(predictions)} samples")

        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict risk probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probability scores (probability of high risk)
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model first.")

        # Apply scaler if available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values if isinstance(X, pd.DataFrame) else X

        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(X_scaled)
            # Return probability of high-risk class (assuming it's the last class)
            return probas[:, -1] if probas.ndim > 1 else probas
        else:
            # Fallback for models without predict_proba
            return self.model.predict(X_scaled)

    def get_risk_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get risk scores with metadata.

        Args:
            X: Feature matrix

        Returns:
            DataFrame with risk_score and risk_level columns
        """
        probas = self.predict_proba(X)

        # Convert probabilities to risk levels
        risk_levels = []
        for p in probas:
            if p >= 0.6:
                risk_levels.append("High")
            elif p >= 0.3:
                risk_levels.append("Medium")
            else:
                risk_levels.append("Low")

        return pd.DataFrame({
            "risk_score": probas,
            "risk_level": risk_levels,
        })

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance from the model.

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            return None

        # Try different importance attributes
        if hasattr(self.model, "feature_importances_"):
            importance = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_).flatten()
        else:
            return None

        if not self.feature_names:
            self.feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({
            "feature": self.feature_names[:len(importance)],
            "importance": importance,
        }).sort_values("importance", ascending=False)
