"""
Risk Extractor Module
=====================

This module extracts and structures risk information from LLM analysis.

It provides data classes and utilities for organizing risk analysis
results into a consistent format.

Example:
    >>> from src.models.llm.risk_extractor import RiskExtractor, RiskAnalysis
    >>> extractor = RiskExtractor()
    >>> analyses = extractor.extract(llm_results)
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class RiskAnalysis:
    """
    Structured risk analysis result.

    This dataclass encapsulates the results of a single project's
    risk analysis, providing a consistent interface for downstream
    processing.

    :ivar project_id: Unique project identifier.
    :vartype project_id: str
    :ivar project_name: Human-readable project name.
    :vartype project_name: str
    :ivar sentiment_score: Sentiment score (-1 to 1).
    :vartype sentiment_score: float
    :ivar sentiment_label: Sentiment label (positive/neutral/negative).
    :vartype sentiment_label: str
    :ivar risk_level: Risk level (high/medium/low).
    :vartype risk_level: str
    :ivar risk_categories: List of detected risk categories.
    :vartype risk_categories: list[str]
    :ivar risk_indicators: Specific risk indicators found.
    :vartype risk_indicators: list[str]
    :ivar key_quotes: Relevant quotes from source text.
    :vartype key_quotes: list[str]
    :ivar confidence: Confidence score (0 to 1).
    :vartype confidence: float
    :ivar summary: One-sentence summary.
    :vartype summary: str

    Example:
        >>> analysis = RiskAnalysis(
        ...     project_id="PROJ-001",
        ...     project_name="Mobile App",
        ...     risk_level="high",
        ...     risk_categories=["technical", "schedule"]
        ... )
    """

    project_id: str
    project_name: str
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    risk_level: str = "medium"
    risk_categories: list[str] = field(default_factory=list)
    risk_indicators: list[str] = field(default_factory=list)
    key_quotes: list[str] = field(default_factory=list)
    confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict:
        """
        Convert to dictionary.

        :return: Dictionary representation of the analysis.
        :rtype: dict
        """
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
            "risk_level": self.risk_level,
            "risk_categories": self.risk_categories,
            "risk_indicators": self.risk_indicators,
            "key_quotes": self.key_quotes,
            "confidence": self.confidence,
            "summary": self.summary,
        }


class RiskExtractor:
    """
    Extract and structure risk information from LLM outputs.

    This class follows the Single Responsibility Principle by focusing
    solely on structuring and normalizing risk analysis results.

    :cvar RISK_CATEGORIES: Standard risk category names.
    :vartype RISK_CATEGORIES: list[str]

    :ivar analyses: List of extracted risk analyses.
    :vartype analyses: list[RiskAnalysis]

    Example:
        >>> extractor = RiskExtractor()
        >>> analyses = extractor.extract(llm_results)
        >>> df = extractor.to_dataframe()
    """

    RISK_CATEGORIES: list[str] = [
        "technical",
        "resource",
        "schedule",
        "scope",
        "budget",
    ]

    def __init__(self) -> None:
        """Initialize the risk extractor."""
        self.analyses: list[RiskAnalysis] = []

    def extract(self, llm_results: list[dict]) -> list[RiskAnalysis]:
        """
        Extract structured risk analyses from LLM results.

        :param llm_results: List of LLM analysis dicts.
        :type llm_results: list[dict]
        :return: List of RiskAnalysis objects.
        :rtype: list[RiskAnalysis]

        Example:
            >>> results = [{"project_id": "P1", "risk_level": "high", ...}]
            >>> analyses = extractor.extract(results)
        """
        self.analyses = []

        for result in llm_results:
            analysis = self._create_analysis(result)
            self.analyses.append(analysis)

        logger.info(f"Extracted {len(self.analyses)} risk analyses")

        return self.analyses

    def _create_analysis(self, result: dict) -> RiskAnalysis:
        """
        Create a RiskAnalysis from a result dict.

        :param result: LLM result dictionary.
        :type result: dict
        :return: RiskAnalysis instance.
        :rtype: RiskAnalysis
        """
        return RiskAnalysis(
            project_id=result.get("project_id", ""),
            project_name=result.get("project_name", "Unknown"),
            sentiment_score=self._normalize_sentiment(result.get("sentiment_score", 0)),
            sentiment_label=result.get("sentiment_label", "neutral"),
            risk_level=self._normalize_risk_level(result.get("risk_level", "medium")),
            risk_categories=self._extract_categories(result.get("risk_categories", [])),
            risk_indicators=result.get("risk_indicators", []),
            key_quotes=result.get("key_quotes", []),
            confidence=result.get("confidence", 0.0),
            summary=result.get("summary", ""),
        )

    def _normalize_sentiment(self, score: float) -> float:
        """
        Normalize sentiment score to -1 to 1 range.

        :param score: Raw sentiment score.
        :type score: float
        :return: Normalized score.
        :rtype: float
        """
        try:
            score = float(score)
            return max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0

    def _normalize_risk_level(self, level: str) -> str:
        """
        Normalize risk level to standard values.

        :param level: Raw risk level string.
        :type level: str
        :return: Normalized risk level.
        :rtype: str
        """
        level = str(level).lower().strip()

        high_indicators = ["high", "critical", "severe"]
        medium_indicators = ["medium", "moderate", "med"]
        low_indicators = ["low", "minimal", "none"]

        if level in high_indicators:
            return "high"
        if level in medium_indicators:
            return "medium"
        if level in low_indicators:
            return "low"

        return "medium"

    def _extract_categories(self, categories: list) -> list[str]:
        """
        Extract and normalize risk categories.

        :param categories: Raw category list.
        :type categories: list
        :return: Normalized category list.
        :rtype: list[str]
        """
        normalized: list[str] = []

        for cat in categories:
            cat_lower = str(cat).lower().strip()
            for standard_cat in self.RISK_CATEGORIES:
                if standard_cat in cat_lower and standard_cat not in normalized:
                    normalized.append(standard_cat)
                    break

        return normalized

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert analyses to DataFrame.

        :return: DataFrame with risk analysis results.
        :rtype: pd.DataFrame

        Example:
            >>> df = extractor.to_dataframe()
            >>> high_risk = df[df["risk_level"] == "high"]
        """
        if not self.analyses:
            return pd.DataFrame()

        data = [a.to_dict() for a in self.analyses]
        df = pd.DataFrame(data)

        # Convert lists to strings for display
        list_columns = ["risk_categories", "risk_indicators", "key_quotes"]
        for col in list_columns:
            if col in df.columns:
                df[f"{col}_str"] = df[col].apply(
                    lambda x: "; ".join(x) if x else ""
                )

        return df

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics from analyses.

        :return: Dict with summary statistics.
        :rtype: dict

        Example:
            >>> stats = extractor.get_summary_stats()
            >>> print(f"High risk count: {stats['risk_distribution']['high']}")
        """
        if not self.analyses:
            return {}

        df = self.to_dataframe()

        return {
            "total_projects": len(self.analyses),
            "avg_sentiment": float(df["sentiment_score"].mean()),
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
            "avg_confidence": float(df["confidence"].mean()),
            "category_counts": self._count_categories(),
        }

    def _count_categories(self) -> dict[str, int]:
        """
        Count occurrences of each risk category.

        :return: Dict mapping categories to counts.
        :rtype: dict[str, int]
        """
        counts = {cat: 0 for cat in self.RISK_CATEGORIES}

        for analysis in self.analyses:
            for cat in analysis.risk_categories:
                if cat in counts:
                    counts[cat] += 1

        return counts

    def get_high_risk_projects(self) -> list[RiskAnalysis]:
        """
        Get all high-risk project analyses.

        :return: List of high-risk analyses.
        :rtype: list[RiskAnalysis]

        Example:
            >>> high_risk = extractor.get_high_risk_projects()
            >>> for proj in high_risk:
            ...     print(proj.project_name)
        """
        return [a for a in self.analyses if a.risk_level == "high"]

    def get_projects_by_category(self, category: str) -> list[RiskAnalysis]:
        """
        Get projects with a specific risk category.

        :param category: Risk category to filter by.
        :type category: str
        :return: List of matching analyses.
        :rtype: list[RiskAnalysis]

        Example:
            >>> technical_risk = extractor.get_projects_by_category("technical")
        """
        category = category.lower()
        return [a for a in self.analyses if category in a.risk_categories]
