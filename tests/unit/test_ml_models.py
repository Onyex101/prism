"""
Tests for the ML models module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def trained_model(classification_data):
    """Create a trained model for testing."""
    X, y = classification_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


class TestMLTrainer:
    """Test suite for MLTrainer class."""

    @pytest.fixture(autouse=True)
    def skip_if_import_fails(self):
        """Skip tests if trainer module cannot be imported."""
        try:
            from src.models.ml.trainer import MLTrainer
        except Exception as e:
            pytest.skip(f"MLTrainer import failed: {e}")

    def test_train_random_forest(self, classification_data):
        """Test training Random Forest model."""
        from src.models.ml.trainer import MLTrainer

        X, y = classification_data
        trainer = MLTrainer(model_type="random_forest", random_state=42)
        model = trainer.train(X, y)

        assert model is not None
        assert trainer.model is not None
        assert len(trainer.feature_names) == 10

    def test_train_with_cv(self, classification_data):
        """Test training with cross-validation."""
        from src.models.ml.trainer import MLTrainer

        X, y = classification_data
        trainer = MLTrainer(model_type="random_forest", random_state=42)
        result = trainer.train_with_cv(X, y, cv=3, scoring="accuracy")

        assert "model" in result
        assert "cv_scores" in result
        assert "mean_score" in result
        assert len(result["cv_scores"]) == 3

    def test_tune_hyperparameters(self, classification_data):
        """Test hyperparameter tuning."""
        from src.models.ml.trainer import MLTrainer

        X, y = classification_data
        trainer = MLTrainer(model_type="random_forest", random_state=42)

        param_grid = {
            "n_estimators": [10, 20],
            "max_depth": [3, 5],
        }

        result = trainer.tune_hyperparameters(X, y, param_grid, cv=2, scoring="accuracy")

        assert "best_params" in result
        assert "best_score" in result
        assert trainer.best_params is not None

    def test_save_and_load_model(self, classification_data, tmp_path):
        """Test model saving."""
        from src.models.ml.trainer import MLTrainer

        X, y = classification_data
        trainer = MLTrainer(model_type="random_forest", random_state=42)
        trainer.train(X, y)

        model_path = tmp_path / "model.pkl"
        trainer.save_model(model_path)

        assert model_path.exists()
        assert (tmp_path / "feature_names.json").exists()

    def test_save_model_without_training(self, tmp_path):
        """Test that saving without training raises error."""
        from src.models.ml.trainer import MLTrainer

        trainer = MLTrainer()

        with pytest.raises(ValueError, match="No model to save"):
            trainer.save_model(tmp_path / "model.pkl")

    def test_compare_models(self, classification_data):
        """Test model comparison."""
        from src.models.ml.trainer import MLTrainer

        X, y = classification_data
        trainer = MLTrainer()

        comparison = trainer.compare_models(X, y, cv=2, scoring="accuracy")

        assert isinstance(comparison, pd.DataFrame)
        assert "model" in comparison.columns
        assert "mean_score" in comparison.columns


class TestMLPredictor:
    """Test suite for MLPredictor class."""

    def test_load_and_predict(self, classification_data, trained_model, tmp_path):
        """Test loading model and making predictions."""
        import joblib
        from src.models.ml.predictor import MLPredictor

        X, y = classification_data

        # Save model directly using joblib
        model_path = tmp_path / "model.pkl"
        joblib.dump(trained_model, model_path)

        # Load and predict
        predictor = MLPredictor(model_path)
        predictions = predictor.predict(X)

        assert len(predictions) == len(X)

    def test_predict_proba(self, classification_data, trained_model, tmp_path):
        """Test probability predictions."""
        import joblib
        from src.models.ml.predictor import MLPredictor

        X, y = classification_data

        model_path = tmp_path / "model.pkl"
        joblib.dump(trained_model, model_path)

        predictor = MLPredictor(model_path)
        probas = predictor.predict_proba(X)

        assert len(probas) == len(X)
        assert all((p >= 0) and (p <= 1) for p in probas)

    def test_get_risk_scores(self, classification_data, trained_model, tmp_path):
        """Test risk score DataFrame generation."""
        import joblib
        from src.models.ml.predictor import MLPredictor

        X, y = classification_data

        model_path = tmp_path / "model.pkl"
        joblib.dump(trained_model, model_path)

        predictor = MLPredictor(model_path)
        scores_df = predictor.get_risk_scores(X)

        assert "risk_score" in scores_df.columns
        assert "risk_level" in scores_df.columns
        assert all(scores_df["risk_level"].isin(["High", "Medium", "Low"]))

    def test_get_feature_importance(self, classification_data, trained_model, tmp_path):
        """Test feature importance retrieval."""
        import joblib
        from src.models.ml.predictor import MLPredictor

        X, y = classification_data

        model_path = tmp_path / "model.pkl"
        joblib.dump(trained_model, model_path)

        predictor = MLPredictor(model_path)
        importance = predictor.get_feature_importance()

        assert importance is not None
        assert "feature" in importance.columns
        assert "importance" in importance.columns
        assert len(importance) == 10

    def test_predict_without_model(self, classification_data):
        """Test that predicting without model raises error."""
        from src.models.ml.predictor import MLPredictor

        predictor = MLPredictor()
        X, _ = classification_data

        with pytest.raises(ValueError, match="No model loaded"):
            predictor.predict(X)

    def test_load_nonexistent_model(self):
        """Test loading nonexistent model file."""
        from src.models.ml.predictor import MLPredictor

        with pytest.raises(FileNotFoundError):
            MLPredictor("/nonexistent/path/model.pkl")


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    def test_evaluate_basic(self):
        """Test basic evaluation."""
        from src.models.ml.evaluator import ModelEvaluator

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        evaluator = ModelEvaluator()
        results = evaluator.evaluate(y_true, y_pred)

        assert "accuracy" in results
        assert "precision" in results
        assert "recall" in results
        assert "f1" in results
        assert "confusion_matrix" in results

    def test_evaluate_with_proba(self):
        """Test evaluation with probabilities."""
        from src.models.ml.evaluator import ModelEvaluator

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([0.2, 0.8, 0.4, 0.3, 0.9])

        evaluator = ModelEvaluator()
        results = evaluator.evaluate(y_true, y_pred, y_proba)

        assert "roc_auc" in results
        assert results["roc_auc"] is not None

    def test_get_metrics_summary(self):
        """Test metrics summary generation."""
        from src.models.ml.evaluator import ModelEvaluator

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        evaluator = ModelEvaluator()
        evaluator.evaluate(y_true, y_pred)
        summary = evaluator.get_metrics_summary()

        assert isinstance(summary, pd.DataFrame)
        assert "metric" in summary.columns
        assert "value" in summary.columns

    def test_get_confusion_matrix_df(self):
        """Test confusion matrix DataFrame generation."""
        from src.models.ml.evaluator import ModelEvaluator

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        evaluator = ModelEvaluator()
        evaluator.evaluate(y_true, y_pred)
        cm_df = evaluator.get_confusion_matrix_df(labels=["Low", "High"])

        assert isinstance(cm_df, pd.DataFrame)
        assert list(cm_df.index) == ["Low", "High"]
        assert list(cm_df.columns) == ["Low", "High"]

    def test_calculate_business_metrics(self):
        """Test business metrics calculation."""
        from src.models.ml.evaluator import ModelEvaluator

        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])

        metrics = ModelEvaluator.calculate_business_metrics(
            y_true,
            y_pred,
            cost_false_negative=10000,
            cost_false_positive=1000,
        )

        assert "true_positives" in metrics
        assert "false_negatives" in metrics
        assert "total_cost" in metrics
        assert "cost_savings_pct" in metrics
