"""
Model Evaluator Module
======================

This module evaluates ML model performance with various metrics.

It provides comprehensive evaluation including classification metrics,
confusion matrices, and business-oriented cost analysis.

Example:
    >>> from src.models.ml.evaluator import ModelEvaluator
    >>> evaluator = ModelEvaluator()
    >>> results = evaluator.evaluate(y_true, y_pred, y_proba)
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """
    Evaluate ML model performance.

    This class follows the Single Responsibility Principle by focusing
    solely on model evaluation. It provides a comprehensive set of
    metrics for classification tasks.

    :ivar results: Dictionary of evaluation results.
    :vartype results: dict

    Example:
        >>> evaluator = ModelEvaluator()
        >>> results = evaluator.evaluate(y_true, y_pred)
        >>> evaluator.print_report()
    """

    def __init__(self) -> None:
        """Initialize the evaluator."""
        self.results: dict = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[list] = None,
    ) -> dict:
        """
        Evaluate model predictions.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        :param y_proba: Predicted probabilities (optional).
        :type y_proba: Optional[np.ndarray]
        :param labels: Class labels for report.
        :type labels: Optional[list]
        :return: Dict with evaluation metrics.
        :rtype: dict

        Example:
            >>> results = evaluator.evaluate(y_test, y_pred, y_proba)
            >>> print(f"F1 Score: {results['f1']:.3f}")
        """
        self.results = self._calculate_basic_metrics(y_true, y_pred)
        self._add_roc_auc(y_true, y_proba)
        self._add_confusion_matrix(y_true, y_pred)
        self._add_classification_report(y_true, y_pred, labels)

        logger.info(f"Evaluation complete. Accuracy: {self.results['accuracy']:.3f}")

        return self.results

    def _calculate_basic_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict:
        """
        Calculate basic classification metrics.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        :return: Dictionary of metrics.
        :rtype: dict
        """
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            ),
            "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        }

    def _add_roc_auc(
        self, y_true: np.ndarray, y_proba: Optional[np.ndarray]
    ) -> None:
        """
        Add ROC-AUC score if probabilities available.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_proba: Predicted probabilities.
        :type y_proba: Optional[np.ndarray]
        """
        if y_proba is None:
            self.results["roc_auc"] = None
            return

        try:
            if y_proba.ndim == 1:
                self.results["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            else:
                self.results["roc_auc"] = float(
                    roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
                )
        except ValueError as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")
            self.results["roc_auc"] = None

    def _add_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> None:
        """
        Add confusion matrix to results.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        """
        self.results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    def _add_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list],
    ) -> None:
        """
        Add classification report to results.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        :param labels: Class labels.
        :type labels: Optional[list]
        """
        self.results["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )

    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of key metrics.

        :return: DataFrame with metric names and values.
        :rtype: pd.DataFrame

        Example:
            >>> summary = evaluator.get_metrics_summary()
            >>> print(summary)
        """
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        data = []

        for metric in metrics:
            value = self.results.get(metric)
            if value is not None:
                data.append({"metric": metric, "value": round(value, 4)})

        return pd.DataFrame(data)

    def get_confusion_matrix_df(
        self,
        labels: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Get confusion matrix as DataFrame.

        :param labels: Class labels for index/columns.
        :type labels: Optional[list]
        :return: Confusion matrix as DataFrame.
        :rtype: pd.DataFrame

        Example:
            >>> cm_df = evaluator.get_confusion_matrix_df(["Low", "Medium", "High"])
        """
        cm = self.results.get("confusion_matrix")
        if cm is None:
            return pd.DataFrame()

        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]

        return pd.DataFrame(cm, index=labels, columns=labels)

    def print_report(self) -> None:
        """
        Print evaluation report to console.

        Example:
            >>> evaluator.evaluate(y_true, y_pred)
            >>> evaluator.print_report()
        """
        print("\n" + "=" * 50)
        print("MODEL EVALUATION REPORT")
        print("=" * 50)

        print("\nKey Metrics:")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            value = self.results.get(metric)
            if value is not None:
                print(f"  {metric.upper()}: {value:.4f}")

        print("\nConfusion Matrix:")
        cm = self.results.get("confusion_matrix")
        if cm is not None:
            print(cm)

        print("\nClassification Report:")
        report = self.results.get("classification_report")
        if report:
            print(report)

        print("=" * 50)

    @staticmethod
    def calculate_business_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        cost_false_negative: float = 10000,
        cost_false_positive: float = 1000,
    ) -> dict:
        """
        Calculate business-oriented metrics.

        This method translates classification errors into business costs,
        useful for communicating model value to stakeholders.

        :param y_true: True labels.
        :type y_true: np.ndarray
        :param y_pred: Predicted labels.
        :type y_pred: np.ndarray
        :param cost_false_negative: Cost of missing a high-risk project.
        :type cost_false_negative: float
        :param cost_false_positive: Cost of false alarm.
        :type cost_false_positive: float
        :return: Dict with business metrics.
        :rtype: dict

        Example:
            >>> metrics = ModelEvaluator.calculate_business_metrics(
            ...     y_true, y_pred,
            ...     cost_false_negative=50000,
            ...     cost_false_positive=5000
            ... )
            >>> print(f"Total Cost: ${metrics['total_cost']:,.0f}")
        """
        cm = confusion_matrix(y_true, y_pred)

        # For binary case
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()

            total_cost = (fn * cost_false_negative) + (fp * cost_false_positive)
            max_cost = len(y_true) * cost_false_negative

            return {
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_cost": float(total_cost),
                "cost_savings_pct": (
                    float((1 - total_cost / max_cost) * 100) if max_cost > 0 else 0
                ),
                "alert_precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                "risk_detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            }

        return {"confusion_matrix": cm.tolist()}
