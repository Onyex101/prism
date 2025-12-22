#!/usr/bin/env python3
"""
Train ML Models Script

Trains and saves ML models for risk prediction.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.data import DataLoader, DataValidator, FeatureEngineer
from src.models.ml import MLTrainer, ModelEvaluator
from sklearn.model_selection import train_test_split


def main():
    """Main training function."""
    logger.info("Starting model training...")

    # Load data
    data_path = Path(__file__).parent.parent / "data" / "raw" / "sample_projects.csv"

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    loader = DataLoader()
    df = loader.load(data_path)

    # Validate data
    validator = DataValidator()
    validation_result = validator.validate(df)

    if not validation_result.is_valid:
        logger.warning(f"Data validation issues: {validation_result.errors}")

    # Feature engineering
    fe = FeatureEngineer()
    df = fe.create_features(df)

    # Prepare features and target
    if "risk_level" in df.columns:
        df["target"] = (df["risk_level"] == "High").astype(int)
    else:
        logger.warning("No risk_level column, creating synthetic target")
        df["target"] = (df["completion_rate"] < 50).astype(int)

    # Select feature columns
    feature_cols = [
        col
        for col in df.columns
        if col
        not in [
            "project_id",
            "project_name",
            "target",
            "risk_level",
            "status_comments",
            "project_description",
            "team_feedback",
            "start_date",
            "planned_end_date",
            "actual_end_date",
            "technology_stack",
            "stakeholder_notes",
        ]
        and df[col].dtype in ["int64", "float64"]
    ]

    logger.info(f"Using {len(feature_cols)} features")

    X = df[feature_cols].fillna(0)
    y = df["target"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Compare models
    logger.info("Comparing models...")
    trainer = MLTrainer()
    comparison = trainer.compare_models(X_train, y_train)
    logger.info(f"\nModel comparison:\n{comparison}")

    # Train best model
    best_model_type = comparison.iloc[0]["model"]
    logger.info(f"Training best model: {best_model_type}")

    trainer = MLTrainer(model_type=best_model_type)
    result = trainer.train_with_cv(X_train, y_train)

    logger.info(f"CV Score: {result['mean_score']:.3f} (+/- {result['std_score']:.3f})")

    # Evaluate on test set
    y_pred = trainer.model.predict(X_test)
    y_proba = (
        trainer.model.predict_proba(X_test)[:, 1]
        if hasattr(trainer.model, "predict_proba")
        else y_pred
    )

    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(y_test, y_pred, y_proba)
    evaluator.print_report()

    # Save model
    model_dir = Path(__file__).parent.parent / "models" / "ml"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.pkl"
    trainer.save_model(model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
