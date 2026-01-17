"""
Models Module
=============

This module provides ML and LLM model implementations for risk prediction.

Subpackages:
    ml: Machine learning models for risk prediction.
    llm: Large Language Model integration for text analysis.

Example:
    >>> from src.models.ml import MLTrainer, MLPredictor
    >>> from src.models.llm import LLMAnalyzer
    >>>
    >>> # ML training
    >>> trainer = MLTrainer(model_type="random_forest")
    >>> trainer.train(X_train, y_train)
    >>>
    >>> # LLM analysis
    >>> analyzer = LLMAnalyzer(api_key="sk-...")
    >>> result = analyzer.analyze_project("Project X", "Status text...")
"""
