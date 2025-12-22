"""
Synthetic Data Generator Module

Generates realistic synthetic project data for testing and training.
"""

import random
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class SyntheticDataGenerator:
    """Generate synthetic project data for testing and training."""

    # Risk-correlated comment templates
    HIGH_RISK_COMMENTS = [
        "Critical issues emerging. {issue1}. Team morale declining. {issue2}. Need immediate intervention.",
        "Project facing significant challenges. {issue1}. Stakeholders expressing concerns about {issue2}.",
        "Multiple blockers identified. {issue1}. Resources stretched thin. {issue2}. Timeline at serious risk.",
        "Severe delays due to {issue1}. Technical debt mounting. {issue2}. Escalation to leadership required.",
        "Crisis mode. {issue1}. Key team member resigned. {issue2}. Recovery plan being developed.",
    ]

    MEDIUM_RISK_COMMENTS = [
        "Some challenges but manageable. {issue1}. Team working on solutions. Monitoring closely.",
        "Minor delays expected. {issue1}. Mitigation steps in place. {positive}.",
        "Progress steady but slower than planned. {issue1}. {positive}. Need to watch carefully.",
        "Mixed results this sprint. {positive} but {issue1}. Adjusting approach for next iteration.",
        "Moderate concerns about {issue1}. {positive}. Team confident but realistic.",
    ]

    LOW_RISK_COMMENTS = [
        "On track. {positive}. Team performing well. No blockers identified.",
        "Excellent progress. {positive}. Ahead of schedule on key deliverables. Stakeholders satisfied.",
        "Smooth execution. {positive}. Quality metrics strong. Team morale high.",
        "All milestones met. {positive}. Budget tracking well. No concerns at this time.",
        "Strong sprint velocity. {positive}. Good collaboration with stakeholders. Confident in timeline.",
    ]

    ISSUES = [
        "unclear requirements",
        "technical debt accumulating",
        "dependency delays",
        "resource constraints",
        "scope creep",
        "integration challenges",
        "testing environment issues",
        "vendor delays",
        "knowledge gaps in team",
        "changing priorities",
        "communication breakdowns",
        "infrastructure problems",
    ]

    POSITIVES = [
        "team collaboration excellent",
        "new features well-received",
        "performance improvements visible",
        "good stakeholder feedback",
        "velocity improving",
        "quality metrics strong",
        "ahead on key deliverables",
        "budget on track",
        "team morale high",
        "no blockers identified",
    ]

    def __init__(self, random_seed: int = 42):
        """
        Initialize the generator.

        Args:
            random_seed: Seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        random.seed(random_seed)

    def generate(
        self,
        n_projects: int = 100,
        risk_distribution: Optional[dict] = None,
        date_range: tuple = ("2023-01-01", "2024-12-31"),
        include_text: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic project data.

        Args:
            n_projects: Number of projects to generate
            risk_distribution: Dict with high/medium/low percentages
            date_range: Tuple of (start_date, end_date) for project dates
            include_text: Whether to generate text comments

        Returns:
            DataFrame with synthetic project data
        """
        if risk_distribution is None:
            risk_distribution = {"high": 0.25, "medium": 0.45, "low": 0.30}

        # Assign risk levels
        risk_levels = self._assign_risk_levels(n_projects, risk_distribution)

        projects = []
        for i in range(n_projects):
            risk = risk_levels[i]
            project = self._generate_project(i + 1, risk, date_range, include_text)
            projects.append(project)

        df = pd.DataFrame(projects)

        logger.info(f"Generated {n_projects} synthetic projects")
        logger.info(f"Risk distribution: {df['risk_level'].value_counts().to_dict()}")

        return df

    def _assign_risk_levels(
        self,
        n_projects: int,
        distribution: dict,
    ) -> list[str]:
        """Assign risk levels based on distribution."""
        levels = []
        for level, pct in distribution.items():
            count = int(n_projects * pct)
            levels.extend([level.capitalize()] * count)

        # Fill remaining with medium
        while len(levels) < n_projects:
            levels.append("Medium")

        random.shuffle(levels)
        return levels[:n_projects]

    def _generate_project(
        self,
        project_num: int,
        risk_level: str,
        date_range: tuple,
        include_text: bool,
    ) -> dict:
        """Generate a single project with correlated attributes."""
        # Base attributes
        project_id = f"PROJ-{project_num:03d}"
        project_name = self._generate_project_name()
        project_type = random.choice([
            "New Development", "Enhancement", "Maintenance",
            "Migration", "Integration", "Research",
        ])
        methodology = random.choice(["Agile", "Waterfall", "Hybrid", "Kanban"])
        priority = self._generate_priority(risk_level)
        department = random.choice([
            "Engineering", "Product", "IT", "Data Science", "Security",
        ])
        client_type = random.choice(["Internal", "External"])

        # Dates
        start_date, planned_end_date = self._generate_dates(date_range)

        # Budget and team (correlated with project type)
        budget = self._generate_budget(project_type)
        team_size = self._generate_team_size(budget)

        # Performance metrics (correlated with risk level)
        metrics = self._generate_performance_metrics(risk_level, budget, team_size)

        # Generate comment
        comment = self._generate_comment(risk_level) if include_text else ""

        return {
            "project_id": project_id,
            "project_name": project_name,
            "project_type": project_type,
            "start_date": start_date,
            "planned_end_date": planned_end_date,
            "actual_end_date": None,
            "budget": budget,
            "spent": metrics["spent"],
            "planned_hours": metrics["planned_hours"],
            "actual_hours": metrics["actual_hours"],
            "team_size": team_size,
            "completion_rate": metrics["completion_rate"],
            "status": "Active",
            "priority": priority,
            "methodology": methodology,
            "department": department,
            "client_type": client_type,
            "technology_stack": self._generate_tech_stack(),
            "complexity_score": self._generate_complexity(risk_level),
            "dependencies": random.randint(0, 10),
            "velocity": metrics["velocity"],
            "defect_rate": metrics["defect_rate"],
            "team_turnover": metrics["team_turnover"],
            "risk_level": risk_level,
            "status_comments": comment,
            "project_description": f"Project to {project_name.lower()}.",
            "team_feedback": self._generate_team_feedback(risk_level),
        }

    def _generate_project_name(self) -> str:
        """Generate a realistic project name."""
        prefixes = [
            "Platform", "Customer", "Mobile", "API", "Data",
            "Cloud", "Security", "Analytics", "Portal", "Integration",
        ]
        actions = [
            "Redesign", "Migration", "Enhancement", "Optimization",
            "Modernization", "Upgrade", "Implementation", "Development",
        ]
        return f"{random.choice(prefixes)} {random.choice(actions)}"

    def _generate_dates(self, date_range: tuple) -> tuple:
        """Generate project start and end dates."""
        start = datetime.strptime(date_range[0], "%Y-%m-%d")
        end = datetime.strptime(date_range[1], "%Y-%m-%d")

        project_start = start + timedelta(days=random.randint(0, (end - start).days - 90))
        duration = random.randint(60, 365)
        project_end = project_start + timedelta(days=duration)

        return project_start.strftime("%Y-%m-%d"), project_end.strftime("%Y-%m-%d")

    def _generate_budget(self, project_type: str) -> float:
        """Generate project budget based on type."""
        base_budgets = {
            "New Development": (150000, 500000),
            "Enhancement": (50000, 200000),
            "Maintenance": (30000, 100000),
            "Migration": (100000, 400000),
            "Integration": (75000, 300000),
            "Research": (50000, 200000),
        }
        min_b, max_b = base_budgets.get(project_type, (50000, 200000))
        return round(random.uniform(min_b, max_b), -3)

    def _generate_team_size(self, budget: float) -> int:
        """Generate team size correlated with budget."""
        base = int(budget / 50000) + 2
        return max(2, min(15, base + random.randint(-2, 2)))

    def _generate_performance_metrics(
        self,
        risk_level: str,
        budget: float,
        team_size: int,
    ) -> dict:
        """Generate performance metrics correlated with risk level."""
        if risk_level == "High":
            completion_rate = random.uniform(30, 70)
            budget_overrun = random.uniform(0.1, 0.5)
            hours_overrun = random.uniform(0.1, 0.4)
            velocity = random.uniform(15, 30)
            defect_rate = random.uniform(0.1, 0.25)
            turnover = random.uniform(0.1, 0.3)
        elif risk_level == "Medium":
            completion_rate = random.uniform(40, 80)
            budget_overrun = random.uniform(-0.1, 0.2)
            hours_overrun = random.uniform(-0.1, 0.2)
            velocity = random.uniform(25, 40)
            defect_rate = random.uniform(0.05, 0.15)
            turnover = random.uniform(0.0, 0.15)
        else:  # Low
            completion_rate = random.uniform(60, 95)
            budget_overrun = random.uniform(-0.2, 0.1)
            hours_overrun = random.uniform(-0.2, 0.1)
            velocity = random.uniform(35, 50)
            defect_rate = random.uniform(0.01, 0.08)
            turnover = random.uniform(0.0, 0.1)

        planned_hours = int(budget / 75)
        actual_hours = int(planned_hours * (1 + hours_overrun))
        spent = round(budget * (1 + budget_overrun), -2)

        return {
            "completion_rate": round(completion_rate, 1),
            "spent": max(0, spent),
            "planned_hours": planned_hours,
            "actual_hours": max(0, actual_hours),
            "velocity": round(velocity, 1),
            "defect_rate": round(defect_rate, 3),
            "team_turnover": round(turnover, 3),
        }

    def _generate_priority(self, risk_level: str) -> str:
        """Generate priority (high-risk projects often high priority)."""
        if risk_level == "High":
            return random.choice(["Critical", "High", "High", "Medium"])
        elif risk_level == "Medium":
            return random.choice(["High", "Medium", "Medium", "Low"])
        else:
            return random.choice(["Medium", "Medium", "Low", "Low"])

    def _generate_complexity(self, risk_level: str) -> int:
        """Generate complexity score correlated with risk."""
        if risk_level == "High":
            return random.randint(6, 10)
        elif risk_level == "Medium":
            return random.randint(4, 8)
        else:
            return random.randint(2, 6)

    def _generate_tech_stack(self) -> str:
        """Generate technology stack."""
        languages = ["Python", "Java", "JavaScript", "TypeScript", "Go", "C#"]
        frameworks = ["React", "Django", "Spring", "Node.js", "FastAPI", ".NET"]
        databases = ["PostgreSQL", "MongoDB", "MySQL", "Redis", "Snowflake"]

        return ", ".join([
            random.choice(languages),
            random.choice(frameworks),
            random.choice(databases),
        ])

    def _generate_comment(self, risk_level: str) -> str:
        """Generate status comment based on risk level."""
        if risk_level == "High":
            template = random.choice(self.HIGH_RISK_COMMENTS)
            issues = random.sample(self.ISSUES, 2)
            return template.format(issue1=issues[0], issue2=issues[1])
        elif risk_level == "Medium":
            template = random.choice(self.MEDIUM_RISK_COMMENTS)
            issue = random.choice(self.ISSUES)
            positive = random.choice(self.POSITIVES)
            return template.format(issue1=issue, positive=positive)
        else:
            template = random.choice(self.LOW_RISK_COMMENTS)
            positive = random.choice(self.POSITIVES)
            return template.format(positive=positive)

    def _generate_team_feedback(self, risk_level: str) -> str:
        """Generate team feedback based on risk level."""
        if risk_level == "High":
            return random.choice([
                "Team stressed and concerned about direction.",
                "Morale declining. Need management support.",
                "Frustration with changing requirements.",
            ])
        elif risk_level == "Medium":
            return random.choice([
                "Team working hard. Some concerns but manageable.",
                "Mixed feelings. Good progress but challenges ahead.",
                "Cautiously optimistic. Need clearer priorities.",
            ])
        else:
            return random.choice([
                "Team engaged and motivated. Good collaboration.",
                "High morale. Proud of progress made.",
                "Excellent team dynamics. Well-functioning unit.",
            ])
