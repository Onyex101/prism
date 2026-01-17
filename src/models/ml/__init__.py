"""
Machine Learning Models Module
==============================

This module provides ML model training, prediction, and evaluation.

Classes:
    BaseModel: Abstract base class for ML models.
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

from src.models.ml.base_model import BaseModel
from src.models.ml.evaluator import ModelEvaluator
from src.models.ml.predictor import MLPredictor

# Try to import trainer, which may fail if XGBoost is not available
try:
    from src.models.ml.trainer import MLTrainer
except Exception:
    # XGBoost may raise XGBoostError if libomp is not installed
    MLTrainer = None  # type: ignore

__all__ = [
    "BaseModel",
    "MLTrainer",
    "MLPredictor",
    "ModelEvaluator",
]
