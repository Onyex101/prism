"""
Machine Learning Models Module
==============================

This module provides ML model training, prediction, and evaluation.

Classes:
    MLTrainer: Train and tune ML models.
    MLPredictor: Load models and make predictions.
    ModelEvaluator: Evaluate model performance.

Note:
    Some model types (e.g., XGBoost) require additional dependencies.
    If XGBoost is not available, Random Forest will be used as fallback.

Example:
    >>> from src.models.ml import MLTrainer, MLPredictor, ModelEvaluator
    >>>
    >>> trainer = MLTrainer(model_type="random_forest")
    >>> trainer.train(X_train, y_train)
    >>>
    >>> predictor = MLPredictor("models/ml/model.pkl")
    >>> predictions = predictor.predict(X_test)
    >>>
    >>> evaluator = ModelEvaluator()
    >>> results = evaluator.evaluate(y_test, predictions)
"""

from src.models.ml.evaluator import ModelEvaluator
from src.models.ml.predictor import MLPredictor

# Optional tree-based deps (e.g. XGBoost) can fail at import; keep predictor/evaluator usable.
try:
    from src.models.ml.trainer import MLTrainer
except (ImportError, OSError):
    # OSError: optional native libs (e.g. tree libraries) may fail to load
    MLTrainer = None  # type: ignore

__all__ = [
    "MLTrainer",
    "MLPredictor",
    "ModelEvaluator",
]
