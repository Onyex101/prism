"""
Pytest Configuration and Fixtures

Shared fixtures for all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_projects_df():
    """Create a sample projects DataFrame for testing."""
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
                "On track with minor issues. Team working well.",
                "Behind schedule due to dependencies. Need more resources.",
                "Good progress. Ahead of schedule on key deliverables.",
            ],
        }
    )


@pytest.fixture
def sample_risk_scores():
    """Sample risk score array for testing."""
    return np.array([0.8, 0.5, 0.3])


@pytest.fixture
def sample_decision_matrix():
    """Sample decision matrix for MCDA testing."""
    return np.array(
        [
            [0.8, 0.6, 0.9, 0.7],
            [0.5, 0.8, 0.6, 0.8],
            [0.3, 0.9, 0.4, 0.9],
        ]
    )


@pytest.fixture
def data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def project_root():
    """Path to project root directory."""
    return Path(__file__).parent.parent
