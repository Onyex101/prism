"""
LLM Models Module
=================

This module provides Large Language Model integration for text analysis
and risk extraction from project documentation.

Classes:
    LLMAnalyzer: Analyze project text using LLMs.
    RiskExtractor: Extract and structure risk information.
    RiskAnalysis: Structured risk analysis result dataclass.

Example:
    >>> from src.models.llm import LLMAnalyzer, RiskExtractor
    >>> analyzer = LLMAnalyzer(api_key="sk-...")
    >>> result = analyzer.analyze_project("Project X", "Status comments...")
    >>>
    >>> extractor = RiskExtractor()
    >>> analyses = extractor.extract([result])
"""

from src.models.llm.analyzer import LLMAnalyzer
from src.models.llm.risk_extractor import RiskAnalysis, RiskExtractor

__all__ = [
    "LLMAnalyzer",
    "RiskExtractor",
    "RiskAnalysis",
]
