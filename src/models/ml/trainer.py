"""
ML Trainer Module
=================

This module handles training and tuning of ML models for risk prediction.

It supports multiple model types (Random Forest, XGBoost, LightGBM) and
provides cross-validation and hyperparameter tuning capabilities.

Example:
    >>> from src.models.ml.trainer import MLTrainer
    >>> trainer = MLTrainer(model_type="random_forest")
    >>> result = trainer.train_with_cv(X, y, cv=5)
"""

import json
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


class MLTrainer:
    """
    Train and tune ML models for risk prediction.

    This class follows the Single Responsibility Principle by focusing
    on model training operations. It supports multiple model types and
    provides a consistent interface.

    :ivar model_type: Type of model to train.
    :vartype model_type: str
    :ivar random_state: Random seed for reproducibility.
    :vartype random_state: int
    :ivar model: Trained model instance.
    :vartype model: Optional[Any]
    :ivar best_params: Best parameters from tuning.
    :vartype best_params: dict
    :ivar cv_scores: Cross-validation scores.
    :vartype cv_scores: list
    :ivar feature_names: Names of features used in training.
    :vartype feature_names: list[str]

    Example:
        >>> trainer = MLTrainer(model_type="xgboost")
        >>> model = trainer.train(X_train, y_train)
        >>> trainer.save_model("models/ml/model.pkl")
    """

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
    ) -> None:
        """
        Initialize the trainer.

        :param model_type: Type of model to train. Options: "random_forest",
            "xgboost", "lightgbm".
        :type model_type: str
        :param random_state: Random seed for reproducibility.
        :type random_state: int
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.best_params: dict = {}
        self.cv_scores: list = []
        self.feature_names: list[str] = []

    def get_model(self, **params: Any) -> Any:
        """
        Get a model instance.

        :param params: Model parameters to override defaults.
        :return: Model instance.
        :rtype: Any

        Example:
            >>> trainer = MLTrainer(model_type="random_forest")
            >>> model = trainer.get_model(n_estimators=200)
        """
        default_params = {"random_state": self.random_state}
        default_params.update(params)

        if self.model_type == "random_forest":
            return RandomForestClassifier(**default_params)

        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                **default_params,
            )

        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(verbose=-1, **default_params)

        logger.warning(
            f"Model type {self.model_type} not available, using RandomForest"
        )
        return RandomForestClassifier(**default_params)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **params: Any,
    ) -> Any:
        """
        Train a model.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :param y: Target labels.
        :type y: pd.Series
        :param params: Model parameters.
        :return: Trained model.
        :rtype: Any

        Example:
            >>> model = trainer.train(X_train, y_train, n_estimators=100)
        """
        self.feature_names = self._extract_feature_names(X)
        self.model = self.get_model(**params)
        self.model.fit(X, y)

        logger.info(f"Trained {self.model_type} model")

        return self.model

    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "roc_auc",
        **params: Any,
    ) -> dict:
        """
        Train with cross-validation.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :param y: Target labels.
        :type y: pd.Series
        :param cv: Number of CV folds.
        :type cv: int
        :param scoring: Scoring metric (e.g., "roc_auc", "f1", "accuracy").
        :type scoring: str
        :param params: Model parameters.
        :return: Dict with model and CV scores.
        :rtype: dict

        Example:
            >>> result = trainer.train_with_cv(X, y, cv=5, scoring="f1")
            >>> print(f"Mean score: {result['mean_score']:.3f}")
        """
        self.feature_names = self._extract_feature_names(X)

        model = self.get_model(**params)
        self.cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring).tolist()

        # Fit on full data
        self.model = self.get_model(**params)
        self.model.fit(X, y)

        mean_score = np.mean(self.cv_scores)
        std_score = np.std(self.cv_scores)

        logger.info(
            f"Trained {self.model_type} with CV. "
            f"Mean {scoring}: {mean_score:.3f} (+/- {std_score:.3f})"
        )

        return {
            "model": self.model,
            "cv_scores": self.cv_scores,
            "mean_score": float(mean_score),
            "std_score": float(std_score),
        }

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: dict,
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> dict:
        """
        Tune hyperparameters using grid search.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :param y: Target labels.
        :type y: pd.Series
        :param param_grid: Parameter grid for search.
        :type param_grid: dict
        :param cv: Number of CV folds.
        :type cv: int
        :param scoring: Scoring metric.
        :type scoring: str
        :return: Dict with best model and parameters.
        :rtype: dict

        Example:
            >>> param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
            >>> result = trainer.tune_hyperparameters(X, y, param_grid)
            >>> print(f"Best params: {result['best_params']}")
        """
        self.feature_names = self._extract_feature_names(X)

        base_model = self.get_model()

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X, y)

        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {grid_search.best_score_:.3f}")

        return {
            "model": self.model,
            "best_params": self.best_params,
            "best_score": float(grid_search.best_score_),
            "cv_results": grid_search.cv_results_,
        }

    def save_model(
        self,
        model_path: Union[str, Path],
        save_features: bool = True,
    ) -> None:
        """
        Save trained model to disk.

        :param model_path: Path to save model.
        :type model_path: Union[str, Path]
        :param save_features: Whether to save feature names.
        :type save_features: bool
        :raises ValueError: If no model to save.

        Example:
            >>> trainer.save_model("models/ml/best_model.pkl")
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")

        if save_features and self.feature_names:
            feature_path = model_path.parent / "feature_names.json"
            with open(feature_path, "w") as f:
                json.dump(self.feature_names, f)

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> pd.DataFrame:
        """
        Compare multiple model types.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :param y: Target labels.
        :type y: pd.Series
        :param cv: Number of CV folds.
        :type cv: int
        :param scoring: Scoring metric.
        :type scoring: str
        :return: DataFrame with comparison results.
        :rtype: pd.DataFrame

        Example:
            >>> comparison = trainer.compare_models(X, y)
            >>> print(comparison)
        """
        model_types = self._get_available_model_types()
        results = []

        for model_type in model_types:
            self.model_type = model_type
            model = self.get_model()

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            results.append(
                {
                    "model": model_type,
                    "mean_score": float(np.mean(scores)),
                    "std_score": float(np.std(scores)),
                    "min_score": float(np.min(scores)),
                    "max_score": float(np.max(scores)),
                }
            )

            logger.info(
                f"{model_type}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})"
            )

        return pd.DataFrame(results).sort_values("mean_score", ascending=False)

    def _extract_feature_names(self, X: pd.DataFrame) -> list[str]:
        """
        Extract feature names from DataFrame.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: List of feature names.
        :rtype: list[str]
        """
        return list(X.columns) if isinstance(X, pd.DataFrame) else []

    def _get_available_model_types(self) -> list[str]:
        """
        Get list of available model types.

        :return: List of model type names.
        :rtype: list[str]
        """
        model_types = ["random_forest"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            model_types.append("lightgbm")
        return model_types
