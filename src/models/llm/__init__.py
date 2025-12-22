"""
LLM Integration Submodule

Handles Large Language Model integration for risk analysis.
"""

from .analyzer import LLMAnalyzer
from .risk_extractor import RiskExtractor

__all__ = ["LLMAnalyzer", "RiskExtractor"]
