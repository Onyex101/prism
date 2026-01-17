"""
Integration tests for the PRISM pipeline.

These tests verify end-to-end workflows from data loading
through to risk ranking.
"""

import numpy as np
import pandas as pd
import pytest


class TestDataPipeline:
    """Integration tests for the data processing pipeline."""

    def test_load_validate_process_pipeline(self, tmp_path):
        """Test full data loading, validation, and processing pipeline."""
        from src.data.loader import DataLoader
        from src.data.validator import DataValidator
        from src.data.preprocessor import DataPreprocessor
        from src.data.feature_engineer import FeatureEngineer

        # Create sample CSV
        df = pd.DataFrame(
            {
                "project_id": ["P1", "P2", "P3"],
                "project_name": ["Alpha", "Beta", "Gamma"],
                "start_date": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "planned_end_date": ["2024-06-01", "2024-07-01", "2024-08-01"],
                "budget": [100000, 150000, 200000],
                "spent": [80000, 160000, 100000],
                "planned_hours": [1000, 1500, 2000],
                "actual_hours": [900, 1600, 1000],
                "team_size": [5, 7, 8],
                "completion_rate": [75.0, 60.0, 45.0],
                "status": ["Active", "Active", "Active"],
                "status_comments": ["Comment 1" * 20, "Comment 2" * 20, "Comment 3" * 20],
            }
        )
        csv_path = tmp_path / "test_projects.csv"
        df.to_csv(csv_path, index=False)

        # Load
        loader = DataLoader()
        loaded_df = loader.load(csv_path)
        assert len(loaded_df) == 3

        # Validate
        validator = DataValidator()
        result = validator.validate(loaded_df)
        assert result.is_valid

        # Create features
        engineer = FeatureEngineer()
        featured_df = engineer.create_features(loaded_df)
        assert "schedule_performance_index" in featured_df.columns
        assert "cost_performance_index" in featured_df.columns

        # Preprocess
        preprocessor = DataPreprocessor()
        numerical_cols = ["budget", "spent", "team_size", "completion_rate"]
        categorical_cols = ["status"]
        processed_df = preprocessor.fit_transform(featured_df, numerical_cols, categorical_cols)
        assert processed_df is not None


class TestMLPipeline:
    """Integration tests for the ML pipeline."""

    def test_train_predict_evaluate_pipeline(self):
        """Test full ML training, prediction, and evaluation pipeline."""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from src.models.ml.evaluator import ModelEvaluator

        # Generate sample data
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y)

        # Split data
        train_size = int(0.8 * len(X_df))
        X_train, X_test = X_df[:train_size], X_df[train_size:]
        y_train, y_test = y_series[:train_size], y_series[train_size:]

        # Train using sklearn directly (avoiding XGBoost import issues)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train, cv=3)
        model.fit(X_train, y_train)

        assert cv_scores.mean() > 0.5

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        evaluator = ModelEvaluator()
        eval_results = evaluator.evaluate(y_test, y_pred, y_proba)

        assert eval_results["accuracy"] > 0.5
        assert eval_results["f1"] > 0.0


class TestMCDAPipeline:
    """Integration tests for the MCDA pipeline."""

    def test_ranking_pipeline(self, sample_projects_df):
        """Test project ranking pipeline."""
        from src.data.feature_engineer import FeatureEngineer
        from src.mcda.ranker import ProjectRanker

        # Create features
        engineer = FeatureEngineer()
        featured_df = engineer.create_features(sample_projects_df)

        # Generate mock ML scores
        ml_scores = pd.Series([0.8, 0.5, 0.2])  # Higher = riskier

        # Rank projects
        ranker = ProjectRanker()
        rankings = ranker.rank(featured_df, ml_scores=ml_scores)

        assert len(rankings) == 3
        assert "rank" in rankings.columns
        assert "risk_level" in rankings.columns
        assert rankings["rank"].tolist() == [1, 2, 3] or set(rankings["rank"]) == {1, 2, 3}

    def test_sensitivity_analysis(self, sample_projects_df):
        """Test sensitivity analysis on rankings."""
        from src.data.feature_engineer import FeatureEngineer
        from src.mcda.ranker import ProjectRanker

        engineer = FeatureEngineer()
        featured_df = engineer.create_features(sample_projects_df)

        ranker = ProjectRanker()
        ranker.rank(featured_df)

        # Run sensitivity analysis
        sensitivity = ranker.sensitivity_analysis(featured_df, weight_variation=0.1)

        assert "criteria_sensitivity" in sensitivity
        assert len(sensitivity["criteria_sensitivity"]) > 0


class TestSyntheticDataPipeline:
    """Integration tests for synthetic data generation and processing."""

    def test_generate_and_process_pipeline(self):
        """Test generating and processing synthetic data."""
        from src.data.generator import SyntheticDataGenerator
        from src.data.validator import DataValidator
        from src.data.feature_engineer import FeatureEngineer

        # Generate
        generator = SyntheticDataGenerator(random_seed=42)
        df = generator.generate(n_projects=50)

        assert len(df) == 50

        # Validate
        validator = DataValidator()
        result = validator.validate(df)
        assert result.is_valid

        # Create features
        engineer = FeatureEngineer()
        featured_df = engineer.create_features(df)

        # Verify risk correlations exist
        high_risk = featured_df[featured_df["risk_level"] == "High"]
        low_risk = featured_df[featured_df["risk_level"] == "Low"]

        # High risk should generally have worse SPI/CPI
        if len(high_risk) > 5 and len(low_risk) > 5:
            assert (
                high_risk["schedule_performance_index"].mean()
                <= low_risk["schedule_performance_index"].mean() + 0.5
            )


class TestVisualizationIntegration:
    """Integration tests for visualization with real data."""

    def test_charts_with_rankings(self, sample_rankings_df):
        """Test creating charts with ranking data."""
        from src.visualization.risk_charts import RiskCharts

        # Test pie chart
        pie_fig = RiskCharts.risk_distribution_pie(sample_rankings_df)
        assert pie_fig is not None

        # Test bar chart
        bar_fig = RiskCharts.risk_score_bar(
            sample_rankings_df,
            top_n=3,
            score_col="mcda_score",
        )
        assert bar_fig is not None

        # Test gauge
        gauge_fig = RiskCharts.risk_gauge(0.35, "Test Project")
        assert gauge_fig is not None

    def test_feature_importance_chart(self, sample_feature_importance_df):
        """Test feature importance chart."""
        from src.visualization.risk_charts import RiskCharts

        fig = RiskCharts.feature_importance_bar(sample_feature_importance_df, top_n=3)
        assert fig is not None
