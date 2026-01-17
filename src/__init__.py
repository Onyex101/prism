"""
PRISM - Predictive Risk Intelligence for Software Management
============================================================

This package provides AI-powered project risk analysis combining
Machine Learning, Large Language Models, and Multi-Criteria Decision Analysis.

Subpackages:
    data: Data loading, validation, preprocessing, and feature engineering.
    models: ML and LLM model implementations.
    mcda: Multi-Criteria Decision Analysis for project ranking.
    explainability: Model interpretation with SHAP.
    chat: Conversational AI assistant.
    visualization: Risk visualization charts.
    utils: Logging and utility functions.

Example:
    >>> from src.data import DataLoader, FeatureEngineer
    >>> from src.models.ml import MLTrainer, MLPredictor
    >>> from src.mcda import ProjectRanker
    >>>
    >>> # Load and process data
    >>> loader = DataLoader()
    >>> df = loader.load("data/projects.csv")
    >>>
    >>> # Create features
    >>> engineer = FeatureEngineer()
    >>> df = engineer.create_features(df)
    >>>
    >>> # Train model
    >>> trainer = MLTrainer(model_type="random_forest")
    >>> trainer.train(X_train, y_train)
"""

__version__ = "1.0.0"
__author__ = "PRISM Team"
