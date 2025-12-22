"""
SHAP Explainer Module

Generates SHAP-based explanations for ML model predictions.
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class SHAPExplainer:
    """Generate SHAP explanations for model predictions."""

    def __init__(self, model: Any, feature_names: Optional[list[str]] = None):
        """
        Initialize the SHAP explainer.

        Args:
            model: Trained ML model
            feature_names: List of feature names
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP package not installed. Run: pip install shap")

        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.Explainer] = None
        self.shap_values: Optional[np.ndarray] = None
        self.base_value: Optional[float] = None

    def fit(self, X: pd.DataFrame) -> "SHAPExplainer":
        """
        Fit the SHAP explainer on training data.

        Args:
            X: Training data for background distribution

        Returns:
            Self for method chaining
        """
        if self.feature_names is None:
            self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None

        # Use TreeExplainer for tree-based models, otherwise KernelExplainer
        model_type = type(self.model).__name__

        if any(t in model_type for t in ["Forest", "Gradient", "XGB", "LGBM", "Tree"]):
            self.explainer = shap.TreeExplainer(self.model)
            logger.info(f"Using TreeExplainer for {model_type}")
        else:
            # Use a subset for KernelExplainer (computationally expensive)
            background = shap.sample(X, min(100, len(X)))
            self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
            logger.info(f"Using KernelExplainer for {model_type}")

        return self

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate SHAP values for given samples.

        Args:
            X: Samples to explain

        Returns:
            Array of SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        self.shap_values = self.explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(self.shap_values, list):
            # For classification, use the positive class (last class)
            self.shap_values = self.shap_values[-1]

        logger.debug(f"Generated SHAP values for {len(X)} samples")

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        Returns:
            DataFrame with feature importance
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first.")

        # Mean absolute SHAP value per feature
        importance = np.abs(self.shap_values).mean(axis=0)

        if self.feature_names:
            names = self.feature_names[: len(importance)]
        else:
            names = [f"Feature_{i}" for i in range(len(importance))]

        return pd.DataFrame(
            {
                "feature": names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)

    def explain_instance(
        self,
        X: pd.DataFrame,
        index: int = 0,
    ) -> dict:
        """
        Get explanation for a single instance.

        Args:
            X: Data containing the instance
            index: Index of instance to explain

        Returns:
            Dict with feature contributions
        """
        if self.shap_values is None:
            self.explain(X)

        instance_shap = self.shap_values[index]

        if self.feature_names:
            names = self.feature_names[: len(instance_shap)]
        else:
            names = [f"Feature_{i}" for i in range(len(instance_shap))]

        contributions = []
        for name, value in zip(names, instance_shap):
            contributions.append(
                {
                    "feature": name,
                    "shap_value": float(value),
                    "direction": "increases risk" if value > 0 else "decreases risk",
                }
            )

        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "contributions": contributions,
            "top_positive": [c for c in contributions if c["shap_value"] > 0][:5],
            "top_negative": [c for c in contributions if c["shap_value"] < 0][:5],
        }

    def get_explanation_text(
        self,
        X: pd.DataFrame,
        index: int = 0,
        top_n: int = 3,
    ) -> str:
        """
        Generate natural language explanation.

        Args:
            X: Data containing the instance
            index: Index of instance to explain
            top_n: Number of top features to include

        Returns:
            Human-readable explanation string
        """
        explanation = self.explain_instance(X, index)

        text_parts = ["This prediction is primarily driven by:"]

        for i, contrib in enumerate(explanation["contributions"][:top_n]):
            feature = contrib["feature"]
            direction = contrib["direction"]
            text_parts.append(f"  {i+1}. {feature} ({direction})")

        return "\n".join(text_parts)
