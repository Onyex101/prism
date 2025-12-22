"""
Model Evaluator Module

Evaluates ML model performance with various metrics.
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
    """Evaluate ML model performance."""

    def __init__(self):
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

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            labels: Class labels for report

        Returns:
            Dict with evaluation metrics
        """
        self.results = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

        # ROC-AUC (if probabilities available)
        if y_proba is not None:
            try:
                if y_proba.ndim == 1:
                    self.results["roc_auc"] = roc_auc_score(y_true, y_proba)
                else:
                    self.results["roc_auc"] = roc_auc_score(
                        y_true, y_proba, multi_class="ovr", average="weighted"
                    )
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                self.results["roc_auc"] = None

        # Confusion matrix
        self.results["confusion_matrix"] = confusion_matrix(y_true, y_pred)

        # Classification report
        self.results["classification_report"] = classification_report(
            y_true, y_pred, labels=labels, zero_division=0
        )

        logger.info(f"Evaluation complete. Accuracy: {self.results['accuracy']:.3f}")

        return self.results

    def get_metrics_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of key metrics.

        Returns:
            DataFrame with metric names and values
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

        Args:
            labels: Class labels for index/columns

        Returns:
            Confusion matrix as DataFrame
        """
        cm = self.results.get("confusion_matrix")
        if cm is None:
            return pd.DataFrame()

        if labels is None:
            labels = [f"Class {i}" for i in range(len(cm))]

        return pd.DataFrame(cm, index=labels, columns=labels)

    def print_report(self) -> None:
        """Print evaluation report to console."""
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

        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_false_negative: Cost of missing a high-risk project
            cost_false_positive: Cost of false alarm

        Returns:
            Dict with business metrics
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
                "total_cost": total_cost,
                "cost_savings_pct": (1 - total_cost / max_cost) * 100 if max_cost > 0 else 0,
                "alert_precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
                "risk_detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0,
            }

        return {"confusion_matrix": cm.tolist()}
