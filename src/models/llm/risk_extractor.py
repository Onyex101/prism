"""
Risk Extractor Module

Extracts and structures risk information from LLM analysis.
"""

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger


@dataclass
class RiskAnalysis:
    """Structured risk analysis result."""

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
        """Convert to dictionary."""
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
    """Extract and structure risk information from LLM outputs."""

    RISK_CATEGORIES = ["technical", "resource", "schedule", "scope", "budget"]

    def __init__(self):
        """Initialize the risk extractor."""
        self.analyses: list[RiskAnalysis] = []

    def extract(self, llm_results: list[dict]) -> list[RiskAnalysis]:
        """
        Extract structured risk analyses from LLM results.

        Args:
            llm_results: List of LLM analysis dicts

        Returns:
            List of RiskAnalysis objects
        """
        self.analyses = []

        for result in llm_results:
            analysis = RiskAnalysis(
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
            self.analyses.append(analysis)

        logger.info(f"Extracted {len(self.analyses)} risk analyses")

        return self.analyses

    def _normalize_sentiment(self, score: float) -> float:
        """Normalize sentiment score to -1 to 1 range."""
        try:
            score = float(score)
            return max(-1.0, min(1.0, score))
        except (ValueError, TypeError):
            return 0.0

    def _normalize_risk_level(self, level: str) -> str:
        """Normalize risk level to standard values."""
        level = str(level).lower().strip()
        if level in ["high", "critical", "severe"]:
            return "high"
        elif level in ["medium", "moderate", "med"]:
            return "medium"
        elif level in ["low", "minimal", "none"]:
            return "low"
        return "medium"

    def _extract_categories(self, categories: list) -> list[str]:
        """Extract and normalize risk categories."""
        normalized = []
        for cat in categories:
            cat_lower = str(cat).lower().strip()
            for standard_cat in self.RISK_CATEGORIES:
                if standard_cat in cat_lower:
                    if standard_cat not in normalized:
                        normalized.append(standard_cat)
                    break
        return normalized

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert analyses to DataFrame.

        Returns:
            DataFrame with risk analysis results
        """
        if not self.analyses:
            return pd.DataFrame()

        data = [a.to_dict() for a in self.analyses]
        df = pd.DataFrame(data)

        # Convert lists to strings for easier display
        for col in ["risk_categories", "risk_indicators", "key_quotes"]:
            if col in df.columns:
                df[f"{col}_str"] = df[col].apply(lambda x: "; ".join(x) if x else "")

        return df

    def get_summary_stats(self) -> dict:
        """
        Get summary statistics from analyses.

        Returns:
            Dict with summary statistics
        """
        if not self.analyses:
            return {}

        df = self.to_dataframe()

        return {
            "total_projects": len(self.analyses),
            "avg_sentiment": df["sentiment_score"].mean(),
            "risk_distribution": df["risk_level"].value_counts().to_dict(),
            "avg_confidence": df["confidence"].mean(),
            "category_counts": self._count_categories(),
        }

    def _count_categories(self) -> dict:
        """Count occurrences of each risk category."""
        counts = {cat: 0 for cat in self.RISK_CATEGORIES}
        for analysis in self.analyses:
            for cat in analysis.risk_categories:
                if cat in counts:
                    counts[cat] += 1
        return counts

    def get_high_risk_projects(self) -> list[RiskAnalysis]:
        """Get all high-risk project analyses."""
        return [a for a in self.analyses if a.risk_level == "high"]

    def get_projects_by_category(self, category: str) -> list[RiskAnalysis]:
        """Get projects with a specific risk category."""
        category = category.lower()
        return [a for a in self.analyses if category in a.risk_categories]
