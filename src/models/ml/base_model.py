"""
Base Model Abstract Class

Defines the interface for all ML models in PRISM.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    def __init__(self, name: str, **kwargs):
        """
        Initialize the base model.

        Args:
            name: Model identifier
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.params = kwargs
        self.model: Optional[Any] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix
            y: Target labels

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Array of predictions
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix

        Returns:
            Array of probability scores
        """
        pass

    def get_params(self) -> dict:
        """Get model parameters."""
        return self.params.copy()

    def set_params(self, **params) -> "BaseModel":
        """Set model parameters."""
        self.params.update(params)
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
