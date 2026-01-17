"""
Explainability Module
=====================

This module provides model interpretation and explanation functionality.

Classes:
    SHAPExplainer: Generate SHAP-based explanations for ML predictions.

Example:
    >>> from src.explainability import SHAPExplainer
    >>> explainer = SHAPExplainer(model, feature_names)
    >>> explainer.fit(X_train)
    >>> importance = explainer.get_feature_importance()
"""

from src.explainability.shap_explainer import SHAPExplainer

__all__ = [
    "SHAPExplainer",
]
