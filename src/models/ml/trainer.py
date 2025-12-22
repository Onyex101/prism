"""
ML Trainer Module

Handles training and tuning of ML models.
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
    """Train and tune ML models for risk prediction."""

    def __init__(
        self,
        model_type: str = "random_forest",
        random_state: int = 42,
    ):
        """
        Initialize the trainer.

        Args:
            model_type: Type of model to train
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.best_params: dict = {}
        self.cv_scores: list = []
        self.feature_names: list[str] = []

    def get_model(self, **params) -> Any:
        """
        Get a model instance.

        Args:
            **params: Model parameters

        Returns:
            Model instance
        """
        default_params = {"random_state": self.random_state}
        default_params.update(params)

        if self.model_type == "random_forest":
            return RandomForestClassifier(**default_params)
        elif self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                **default_params,
            )
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            return LGBMClassifier(verbose=-1, **default_params)
        else:
            logger.warning(f"Model type {self.model_type} not available, using RandomForest")
            return RandomForestClassifier(**default_params)

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        **params,
    ) -> Any:
        """
        Train a model.

        Args:
            X: Feature matrix
            y: Target labels
            **params: Model parameters

        Returns:
            Trained model
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

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
        **params,
    ) -> dict:
        """
        Train with cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of CV folds
            scoring: Scoring metric
            **params: Model parameters

        Returns:
            Dict with model and CV scores
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

        model = self.get_model(**params)
        self.cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        # Fit on full data
        self.model = self.get_model(**params)
        self.model.fit(X, y)

        logger.info(
            f"Trained {self.model_type} with CV. "
            f"Mean {scoring}: {np.mean(self.cv_scores):.3f} "
            f"(+/- {np.std(self.cv_scores):.3f})"
        )

        return {
            "model": self.model,
            "cv_scores": self.cv_scores,
            "mean_score": np.mean(self.cv_scores),
            "std_score": np.std(self.cv_scores),
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

        Args:
            X: Feature matrix
            y: Target labels
            param_grid: Parameter grid for search
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Dict with best model and parameters
        """
        self.feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else []

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
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_,
        }

    def save_model(
        self,
        model_path: Union[str, Path],
        save_features: bool = True,
    ) -> None:
        """
        Save trained model to disk.

        Args:
            model_path: Path to save model
            save_features: Whether to save feature names
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

        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            DataFrame with comparison results
        """
        model_types = ["random_forest"]
        if XGBOOST_AVAILABLE:
            model_types.append("xgboost")
        if LIGHTGBM_AVAILABLE:
            model_types.append("lightgbm")

        results = []

        for model_type in model_types:
            self.model_type = model_type
            model = self.get_model()

            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

            results.append({
                "model": model_type,
                "mean_score": np.mean(scores),
                "std_score": np.std(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores),
            })

            logger.info(f"{model_type}: {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")

        return pd.DataFrame(results).sort_values("mean_score", ascending=False)
