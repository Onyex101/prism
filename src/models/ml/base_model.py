"""
Base Model Abstract Class
=========================

This module defines the abstract interface for all ML models in PRISM.

It follows the Liskov Substitution Principle (LSP) by providing a consistent
interface that all model implementations must follow.

Example:
    >>> from src.models.ml.base_model import BaseModel
    >>> class MyModel(BaseModel):
    ...     def fit(self, X, y): ...
    ...     def predict(self, X): ...
    ...     def predict_proba(self, X): ...
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all ML models.

    This class defines the interface that all ML models must implement,
    following the Open/Closed Principle - models are open for extension
    but closed for modification.

    :ivar name: Model identifier.
    :vartype name: str
    :ivar params: Model-specific parameters.
    :vartype params: dict
    :ivar model: Underlying model instance.
    :vartype model: Optional[Any]
    :ivar is_fitted: Whether the model has been fitted.
    :vartype is_fitted: bool

    Example:
        >>> class RandomForestModel(BaseModel):
        ...     def fit(self, X, y):
        ...         self.model = RandomForestClassifier(**self.params)
        ...         self.model.fit(X, y)
        ...         self.is_fitted = True
        ...         return self
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """
        Initialize the base model.

        :param name: Model identifier.
        :type name: str
        :param kwargs: Model-specific parameters.
        """
        self.name = name
        self.params = kwargs
        self.model: Optional[Any] = None
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """
        Fit the model to training data.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :param y: Target labels.
        :type y: pd.Series
        :return: Self for method chaining.
        :rtype: BaseModel
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: Array of predictions.
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.

        :param X: Feature matrix.
        :type X: pd.DataFrame
        :return: Array of probability scores.
        :rtype: np.ndarray
        """
        pass

    def get_params(self) -> dict:
        """
        Get model parameters.

        :return: Copy of model parameters.
        :rtype: dict
        """
        return self.params.copy()

    def set_params(self, **params: Any) -> "BaseModel":
        """
        Set model parameters.

        :param params: Parameters to set.
        :return: Self for method chaining.
        :rtype: BaseModel
        """
        self.params.update(params)
        return self

    def __repr__(self) -> str:
        """
        Return string representation of the model.

        :return: String representation.
        :rtype: str
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
