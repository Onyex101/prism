"""
SHAP Explainer Module
=====================

This module generates SHAP-based explanations for ML model predictions.

SHAP (SHapley Additive exPlanations) values provide interpretable
explanations for individual predictions by measuring feature contributions.

Example:
    >>> from src.explainability.shap_explainer import SHAPExplainer
    >>> explainer = SHAPExplainer(model, feature_names)
    >>> explainer.fit(X_train)
    >>> importance = explainer.get_feature_importance()

Reference:
    Lundberg, S.M., Lee, S.I. (2017). A Unified Approach to Interpreting
    Model Predictions. NIPS 2017.
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
    """
    Generate SHAP explanations for model predictions.

    This class provides model-agnostic explanations using SHAP values.
    It automatically selects the appropriate explainer type based on
    the model architecture.

    :ivar model: Trained ML model.
    :vartype model: Any
    :ivar feature_names: List of feature names.
    :vartype feature_names: Optional[list[str]]
    :ivar explainer: SHAP explainer instance.
    :vartype explainer: Optional[shap.Explainer]
    :ivar shap_values: Computed SHAP values.
    :vartype shap_values: Optional[np.ndarray]
    :ivar base_value: Base value for explanations.
    :vartype base_value: Optional[float]

    Example:
        >>> explainer = SHAPExplainer(trained_model, ["feature_a", "feature_b"])
        >>> explainer.fit(X_train)
        >>> explanation = explainer.explain_instance(X_test, index=0)
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize the SHAP explainer.

        :param model: Trained ML model.
        :type model: Any
        :param feature_names: List of feature names.
        :type feature_names: Optional[list[str]]
        :raises ImportError: If SHAP package not installed.
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

        :param X: Training data for background distribution.
        :type X: pd.DataFrame
        :return: Self for method chaining.
        :rtype: SHAPExplainer

        Example:
            >>> explainer.fit(X_train)
        """
        if self.feature_names is None:
            self.feature_names = self._extract_feature_names(X)

        self.explainer = self._create_explainer(X)

        return self

    def _extract_feature_names(self, X: pd.DataFrame) -> Optional[list[str]]:
        """
        Extract feature names from DataFrame.

        :param X: DataFrame.
        :type X: pd.DataFrame
        :return: List of feature names or None.
        :rtype: Optional[list[str]]
        """
        return list(X.columns) if isinstance(X, pd.DataFrame) else None

    def _create_explainer(self, X: pd.DataFrame) -> shap.Explainer:
        """
        Create appropriate SHAP explainer based on model type.

        :param X: Background data.
        :type X: pd.DataFrame
        :return: SHAP explainer instance.
        :rtype: shap.Explainer
        """
        model_type = type(self.model).__name__

        tree_models = ["Forest", "Gradient", "XGB", "LGBM", "Tree"]
        if any(t in model_type for t in tree_models):
            logger.info(f"Using TreeExplainer for {model_type}")
            return shap.TreeExplainer(self.model)

        # Use KernelExplainer for other models
        background = shap.sample(X, min(100, len(X)))
        logger.info(f"Using KernelExplainer for {model_type}")
        return shap.KernelExplainer(self.model.predict_proba, background)

    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate SHAP values for given samples.

        :param X: Samples to explain.
        :type X: pd.DataFrame
        :return: Array of SHAP values.
        :rtype: np.ndarray
        :raises ValueError: If explainer not fitted.

        Example:
            >>> shap_values = explainer.explain(X_test)
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")

        self.shap_values = self.explainer.shap_values(X)

        # Handle multi-class output
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[-1]

        logger.debug(f"Generated SHAP values for {len(X)} samples")

        return self.shap_values

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get global feature importance from SHAP values.

        :return: DataFrame with feature importance.
        :rtype: pd.DataFrame
        :raises ValueError: If no SHAP values computed.

        Example:
            >>> importance_df = explainer.get_feature_importance()
            >>> print(importance_df.head(10))
        """
        if self.shap_values is None:
            raise ValueError("No SHAP values. Call explain() first.")

        importance = np.abs(self.shap_values).mean(axis=0)
        names = self._get_feature_names(len(importance))

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

        :param X: Data containing the instance.
        :type X: pd.DataFrame
        :param index: Index of instance to explain.
        :type index: int
        :return: Dict with feature contributions.
        :rtype: dict

        Example:
            >>> explanation = explainer.explain_instance(X_test, index=5)
            >>> for contrib in explanation["top_positive"][:3]:
            ...     print(f"{contrib['feature']}: {contrib['shap_value']:.3f}")
        """
        if self.shap_values is None:
            self.explain(X)

        instance_shap = self.shap_values[index]
        names = self._get_feature_names(len(instance_shap))

        contributions = self._build_contributions(names, instance_shap)
        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return {
            "contributions": contributions,
            "top_positive": [c for c in contributions if c["shap_value"] > 0][:5],
            "top_negative": [c for c in contributions if c["shap_value"] < 0][:5],
        }

    def _build_contributions(self, names: list[str], shap_values: np.ndarray) -> list[dict]:
        """
        Build contribution list from names and values.

        :param names: Feature names.
        :type names: list[str]
        :param shap_values: SHAP values for instance.
        :type shap_values: np.ndarray
        :return: List of contribution dicts.
        :rtype: list[dict]
        """
        return [
            {
                "feature": name,
                "shap_value": float(value),
                "direction": "increases risk" if value > 0 else "decreases risk",
            }
            for name, value in zip(names, shap_values)
        ]

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
        return [f"Feature_{i}" for i in range(n_features)]

    def get_explanation_text(
        self,
        X: pd.DataFrame,
        index: int = 0,
        top_n: int = 3,
    ) -> str:
        """
        Generate natural language explanation.

        :param X: Data containing the instance.
        :type X: pd.DataFrame
        :param index: Index of instance to explain.
        :type index: int
        :param top_n: Number of top features to include.
        :type top_n: int
        :return: Human-readable explanation string.
        :rtype: str

        Example:
            >>> explanation_text = explainer.get_explanation_text(X, index=0)
            >>> print(explanation_text)
        """
        explanation = self.explain_instance(X, index)

        text_parts = ["This prediction is primarily driven by:"]

        for i, contrib in enumerate(explanation["contributions"][:top_n]):
            feature = contrib["feature"]
            direction = contrib["direction"]
            text_parts.append(f"  {i + 1}. {feature} ({direction})")

        return "\n".join(text_parts)
