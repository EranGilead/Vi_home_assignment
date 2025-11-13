#!/usr/bin/env python
"""Train interpretable churn models on engineered features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


ModelFactory = Callable[[], Any]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the training script.

    Returns:
        argparse.Namespace: Parsed arguments with paths and model config.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-path",
        default="feature_outputs/features.csv",
        type=Path,
        help="CSV produced by src/feature_plan.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_outputs",
        type=Path,
        help="Directory for trained models and evaluation artifacts.",
    )
    parser.add_argument(
        "--model",
        default="logistic_regression",
        choices=["logistic_regression", "random_forest", "xgboost"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--task",
        default="churn",
        choices=["churn", "benefit"],
        help="Training objective: churn prediction or outreach-benefit ranking.",
    )
    parser.add_argument(
        "--test-size",
        default=0.2,
        type=float,
        help="Holdout fraction for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        default=42,
        type=int,
        help="Random seed for train/test split and model training.",
    )
    parser.add_argument(
        "--rf-n-estimators",
        type=int,
        default=500,
        help="Number of trees for random forest.",
    )
    parser.add_argument(
        "--rf-max-depth",
        type=int,
        default=None,
        help="Optional maximum depth for random forest.",
    )
    parser.add_argument(
        "--xgb-n-estimators",
        type=int,
        default=800,
        help="Boosting rounds for XGBoost.",
    )
    parser.add_argument(
        "--xgb-learning-rate",
        type=float,
        default=0.05,
        help="Learning rate for XGBoost.",
    )
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=4,
        help="Tree depth for XGBoost.",
    )
    parser.add_argument(
        "--xgb-subsample",
        type=float,
        default=0.8,
        help="Row subsample ratio for XGBoost.",
    )
    parser.add_argument(
        "--xgb-colsample",
        type=float,
        default=0.8,
        help="Column subsample ratio for XGBoost.",
    )
    return parser.parse_args()


def load_features(features_path: Path) -> pd.DataFrame:
    """Load engineered features and drop redundant columns if present.

    Args:
        features_path (Path): Path to CSV with engineered features.

    Returns:
        pd.DataFrame: Feature table without redundant columns.
    """
    df = pd.read_csv(features_path)
    redundant_cols = {"claims_icd_diversity"}
    existing = list(redundant_cols.intersection(df.columns))
    if existing:
        df = df.drop(columns=existing)
    return df


def build_model_registry(args, scale_pos_weight: float) -> Dict[str, ModelFactory]:
    """Define supported models.

    Args:
        args: CLI arguments.
        scale_pos_weight (float): Ratio for boosting models (negative/positive).

    Returns:
        dict[str, ModelFactory]: Mapping from model name to factory function.
    """
    return {
        "logistic_regression": lambda: Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=args.random_state,
                    ),
                ),
            ]
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        ),
        "xgboost": lambda: XGBClassifier(
            n_estimators=args.xgb_n_estimators,
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=args.random_state,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
        ),
    }


def evaluate_model(model: Pipeline, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute evaluation metrics for the given model."""
    preds = model.predict(X_test)
    probas = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probas),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    report = classification_report(y_test, preds, digits=4)
    return metrics, report


def serialize_artifacts(
    model: Pipeline,
    metrics: dict,
    report: str,
    output_dir: Path,
    model_name: str,
) -> None:
    """Persist trained model and evaluation artifacts to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{model_name}.joblib"
    metrics_path = output_dir / f"{model_name}_metrics.json"
    report_path = output_dir / f"{model_name}_report.txt"

    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    report_path.write_text(report)


def main() -> None:
    """Entry point for the training workflow."""
    args = parse_args()
    df = load_features(args.features_path)

    drop_cols = {"member_id", "churn"}
    if args.task == "benefit":
        drop_cols.update({"outreach", "outreach_x_low_engagement"})
    feature_cols = [col for col in df.columns if col not in drop_cols]
    X = df[feature_cols].fillna(0)
    if args.task == "churn":
        y = df["churn"].values
    else:
        benefit = ((df["outreach"] == 1) & (df["churn"] == 0)) | (
            (df["outreach"] == 0) & (df["churn"] == 1)
        )
        y = benefit.astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = (neg_count / pos_count) if pos_count else 1.0

    registry = build_model_registry(args, scale_pos_weight)
    if args.model not in registry:
        raise ValueError(f"Unsupported model: {args.model}")
    model = registry[args.model]()
    model.fit(X_train, y_train)

    metrics, report = evaluate_model(model, X_test, y_test)
    metrics["task"] = args.task
    serialize_artifacts(model, metrics, report, args.output_dir, args.model)

    print(f"Model trained: {args.model}")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
