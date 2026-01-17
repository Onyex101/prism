"""
Pytest Configuration and Fixtures
=================================

Shared fixtures for all test modules.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_projects_df():
    """
    Create a sample projects DataFrame for testing.

    :return: DataFrame with sample project data.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(
        {
            "project_id": ["PROJ-001", "PROJ-002", "PROJ-003"],
            "project_name": ["Project Alpha", "Project Beta", "Project Gamma"],
            "start_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
            "planned_end_date": ["2024-06-01", "2024-08-01", "2024-09-01"],
            "budget": [100000, 150000, 200000],
            "spent": [80000, 160000, 100000],
            "planned_hours": [1000, 1500, 2000],
            "actual_hours": [900, 1600, 1000],
            "team_size": [5, 7, 8],
            "completion_rate": [75.0, 60.0, 45.0],
            "status": ["Active", "Active", "Active"],
            "priority": ["High", "Medium", "High"],
            "status_comments": [
                "On track with minor issues. Team working well together.",
                "Behind schedule due to dependencies. Need more resources.",
                "Good progress. Ahead of schedule on key deliverables.",
            ],
        }
    )


@pytest.fixture
def sample_projects_with_risk_df(sample_projects_df):
    """
    Create a sample projects DataFrame with risk level.

    :param sample_projects_df: Base projects DataFrame.
    :return: DataFrame with risk_level column added.
    :rtype: pd.DataFrame
    """
    df = sample_projects_df.copy()
    df["risk_level"] = ["High", "Medium", "Low"]
    return df


@pytest.fixture
def sample_risk_scores():
    """
    Sample risk score array for testing.

    :return: Array of risk scores.
    :rtype: np.ndarray
    """
    return np.array([0.8, 0.5, 0.3])


@pytest.fixture
def sample_decision_matrix():
    """
    Sample decision matrix for MCDA testing.

    :return: Decision matrix array.
    :rtype: np.ndarray
    """
    return np.array(
        [
            [0.8, 0.6, 0.9, 0.7],
            [0.5, 0.8, 0.6, 0.8],
            [0.3, 0.9, 0.4, 0.9],
        ]
    )


@pytest.fixture
def sample_llm_results():
    """
    Sample LLM analysis results for testing.

    :return: List of LLM result dictionaries.
    :rtype: list[dict]
    """
    return [
        {
            "project_id": "PROJ-001",
            "project_name": "Project Alpha",
            "sentiment_score": -0.3,
            "sentiment_label": "negative",
            "risk_level": "high",
            "risk_categories": ["technical", "schedule"],
            "risk_indicators": ["code quality issues", "deadline pressure"],
            "key_quotes": ["facing significant challenges"],
            "confidence": 0.8,
            "summary": "Project has technical and schedule risks.",
        },
        {
            "project_id": "PROJ-002",
            "project_name": "Project Beta",
            "sentiment_score": 0.5,
            "sentiment_label": "positive",
            "risk_level": "low",
            "risk_categories": [],
            "risk_indicators": [],
            "key_quotes": ["on track"],
            "confidence": 0.9,
            "summary": "Project is on track.",
        },
    ]


@pytest.fixture
def data_dir():
    """
    Path to test data directory.

    :return: Path to test_data directory.
    :rtype: Path
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture
def project_root():
    """
    Path to project root directory.

    :return: Path to project root.
    :rtype: Path
    """
    return Path(__file__).parent.parent


@pytest.fixture
def sample_rankings_df():
    """
    Sample rankings DataFrame for testing.

    :return: DataFrame with ranking data.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(
        {
            "project_id": ["PROJ-001", "PROJ-002", "PROJ-003"],
            "project_name": ["Project Alpha", "Project Beta", "Project Gamma"],
            "mcda_score": [0.35, 0.55, 0.80],
            "rank": [1, 2, 3],
            "risk_level": ["High", "Medium", "Low"],
        }
    )


@pytest.fixture
def sample_feature_importance_df():
    """
    Sample feature importance DataFrame for testing.

    :return: DataFrame with feature importance data.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(
        {
            "feature": ["completion_rate", "budget_variance", "team_turnover"],
            "importance": [0.35, 0.25, 0.15],
        }
    )
