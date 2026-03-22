#!/usr/bin/env python3
"""
System Evaluation Script

Runs full evaluation of the PRISM hybrid system: ML predictions, LLM analysis (optional),
MCDA ranking, and comparison of hybrid vs individual approaches.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger


def load_data(data_path: Path) -> pd.DataFrame:
    """Load and prepare project data."""
    from src.data import DataLoader, FeatureEngineer

    loader = DataLoader()
    df = loader.load(data_path)
    fe = FeatureEngineer()
    return fe.create_features(df)


def run_ml_evaluation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """Run ML predictions and evaluation. Returns (df with risk_score, eval_results)."""
    from sklearn.model_selection import train_test_split

    from src.data import FeatureEngineer
    from src.models.ml import MLTrainer, ModelEvaluator

    exclude = [
        "project_id",
        "project_name",
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
    feature_cols = [
        c
        for c in df.columns
        if c not in exclude and df[c].dtype in ["int64", "float64", "int32", "float32"]
    ]
    X = df[feature_cols].fillna(0)
    y = (
        (df["risk_level"] == "High").astype(int)
        if "risk_level" in df.columns
        else (df["completion_rate"] < 50).astype(int)
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if MLTrainer is None:
        logger.warning("MLTrainer unavailable. Skipping ML evaluation.")
        return df, None

    trainer = MLTrainer(model_type="random_forest")
    trainer.train(X_train, y_train)

    y_pred = trainer.model.predict(X_test)
    y_proba = (
        trainer.model.predict_proba(X_test)[:, -1]
        if hasattr(trainer.model, "predict_proba")
        else y_pred
    )

    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(y_test, y_pred, y_proba)

    df = df.copy()
    probas = (
        trainer.model.predict_proba(X)[:, -1]
        if hasattr(trainer.model, "predict_proba")
        else trainer.model.predict(X)
    )
    df["risk_score"] = probas
    df["risk_level_ml"] = ["High" if p >= 0.6 else "Medium" if p >= 0.3 else "Low" for p in probas]

    return df, eval_results


def run_llm_analysis(df: pd.DataFrame, api_key: str | None, max_projects: int = 5) -> pd.DataFrame:
    """Run LLM analysis if API key and status_comments available."""
    if not api_key or "status_comments" not in df.columns:
        return df

    try:
        from src.models.llm import LLMAnalyzer
        from src.models.llm.risk_extractor import RiskExtractor

        analyzer = LLMAnalyzer(api_key=api_key, model="gpt-3.5-turbo")
        projects = df.head(max_projects).to_dict(orient="records")
        results = analyzer.analyze_batch(
            projects, text_field="status_comments", name_field="project_name"
        )

        extractor = RiskExtractor()
        extractor.extract(results)
        llm_df = extractor.to_dataframe()

        df = df.copy()
        merged = df.merge(
            llm_df[["project_name", "sentiment_score", "sentiment_label"]],
            on="project_name",
            how="left",
        )
        df["sentiment_score"] = merged["sentiment_score"]
        df["sentiment_label"] = merged["sentiment_label"]
        logger.info(f"LLM analyzed {len(results)} projects")
    except Exception as e:
        logger.warning(f"LLM analysis failed: {e}")

    return df


def run_mcda_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Run MCDA ranking."""
    from src.mcda import ProjectRanker

    if "project_id" not in df.columns:
        df = df.copy()
        df["project_id"] = df["project_name"] if "project_name" in df.columns else range(len(df))

    ranker = ProjectRanker()
    rankings = ranker.rank(df)

    if "risk_level" in df.columns:
        df = df.drop(columns=["risk_level"])
    df = df.merge(
        rankings[["project_id", "mcda_score", "rank", "risk_level"]], on="project_id", how="left"
    )
    return df


def generate_report(
    df: pd.DataFrame,
    eval_results: dict | None,
    output_path: Path,
) -> None:
    """Generate evaluation report."""
    lines = ["# PRISM System Evaluation Report", ""]

    lines.append("## 1. ML Evaluation")
    if eval_results:
        lines.append(f"- ROC-AUC: {eval_results.get('roc_auc', 0):.3f}")
        lines.append(f"- F1 Score: {eval_results.get('f1', 0):.3f}")
        lines.append(f"- Accuracy: {eval_results.get('accuracy', 0):.3f}")
    else:
        lines.append("- ML evaluation not run (MLTrainer unavailable)")

    lines.append("")
    lines.append("## 2. Risk Distribution")
    if "risk_level" in df.columns:
        dist = df["risk_level"].value_counts()
        for level, count in dist.items():
            lines.append(f"- {level}: {count}")

    lines.append("")
    lines.append("## 3. Top 5 High-Risk Projects (MCDA)")
    if "mcda_score" in df.columns:
        top5 = df.nsmallest(5, "mcda_score")[["project_name", "mcda_score", "rank", "risk_level"]]
        lines.append(top5.to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report saved to {output_path}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate PRISM hybrid system")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/jira_projects.csv"),
        help="Path to input data (default: processed Jira CSV)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/evaluation_report.md"),
        help="Report output path",
    )
    parser.add_argument(
        "--api-key", type=str, default=None, help="OpenAI API key for LLM (optional)"
    )
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM analysis")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_path = project_root / args.data if not args.data.is_absolute() else args.data
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    logger.info("Starting system evaluation...")

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    df = load_data(data_path)
    logger.info(f"Loaded {len(df)} projects")

    df, eval_results = run_ml_evaluation(df)
    logger.info("ML evaluation complete")

    if not args.no_llm:
        api_key = args.api_key or __import__("os").environ.get("OPENAI_API_KEY")
        df = run_llm_analysis(df, api_key, max_projects=5)

    df = run_mcda_ranking(df)
    logger.info("MCDA ranking complete")

    generate_report(df, eval_results, output_path)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
