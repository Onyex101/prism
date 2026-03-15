"""
Tests for the SHAPExplainer module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def trained_model_and_data():
    """Create trained model and data for SHAP testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(50, 5),
        columns=["f1", "f2", "f3", "f4", "f5"],
    )
    y = (X["f1"] + X["f2"] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X


class TestSHAPExplainer:
    """Test suite for SHAPExplainer."""

    @pytest.fixture
    def skip_if_no_shap(self):
        """Skip if SHAP not installed."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

    def test_fit_and_explain(self, trained_model_and_data):
        """Test fit and explain methods."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

        from src.explainability.shap_explainer import SHAPExplainer

        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.fit(X)

        shap_values = explainer.explain(X)
        assert shap_values is not None
        assert shap_values.shape == (len(X), 5)

    def test_get_feature_importance(self, trained_model_and_data):
        """Test get_feature_importance."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

        from src.explainability.shap_explainer import SHAPExplainer

        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.fit(X)
        explainer.explain(X)

        importance_df = explainer.get_feature_importance()
        assert len(importance_df) == 5
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert importance_df["importance"].sum() > 0

    def test_explain_instance(self, trained_model_and_data):
        """Test explain_instance."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

        from src.explainability.shap_explainer import SHAPExplainer

        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.fit(X)

        explanation = explainer.explain_instance(X, index=0)
        assert "contributions" in explanation
        assert "top_positive" in explanation
        assert "top_negative" in explanation
        assert len(explanation["contributions"]) == 5

    def test_get_explanation_text(self, trained_model_and_data):
        """Test get_explanation_text."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

        from src.explainability.shap_explainer import SHAPExplainer

        model, X = trained_model_and_data
        explainer = SHAPExplainer(model, list(X.columns))
        explainer.fit(X)

        text = explainer.get_explanation_text(X, index=0, top_n=3)
        assert "driven by" in text.lower() or "feature" in text.lower()
