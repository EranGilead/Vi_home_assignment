#!/usr/bin/env python
"""Evaluate a trained churn model on a provided feature dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="model_outputs/logistic_regression.joblib",
        type=Path,
        help="Path to a trained sklearn pipeline with predict_proba.",
    )
    parser.add_argument(
        "--features-path",
        required=True,
        type=Path,
        help="CSV of engineered features (must contain 'churn').",
    )
    parser.add_argument(
        "--output-dir",
        default="model_outputs",
        type=Path,
        help="Directory to store evaluation artifacts.",
    )
    parser.add_argument(
        "--split-name",
        default="test",
        type=str,
        help="Identifier for the evaluated split (used in filenames).",
    )
    parser.add_argument(
        "--task",
        default="churn",
        choices=["churn", "benefit"],
        help="Match evaluation to training objective.",
    )
    return parser.parse_args()


def load_features(features_path: Path) -> pd.DataFrame:
    """Load engineered features and drop redundant columns if present."""
    df = pd.read_csv(features_path)
    redundant_cols = {"claims_icd_diversity"}
    df = df.drop(columns=[c for c in redundant_cols if c in df.columns])
    if "churn" not in df.columns:
        raise ValueError("Features file must contain 'churn' column for evaluation.")
    return df


def evaluate(model, X: np.ndarray, y: np.ndarray) -> tuple[dict, str]:
    """Compute standard classification metrics."""
    preds = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y, preds),
        "precision": precision_score(y, preds),
        "recall": recall_score(y, preds),
        "f1": f1_score(y, preds),
        "roc_auc": roc_auc_score(y, probas),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    report = classification_report(y, preds, digits=4)
    return metrics, report


def main() -> None:
    args = parse_args()
    df = load_features(args.features_path)
    drop_cols = {"member_id", "churn"}
    if args.task == "benefit":
        drop_cols.update({"outreach", "outreach_x_low_engagement"})
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)
    if args.task == "churn":
        y = df["churn"].values
    else:
        benefit = ((df["outreach"] == 1) & (df["churn"] == 0)) | (
            (df["outreach"] == 0) & (df["churn"] == 1)
        )
        y = benefit.astype(int).values

    model = joblib.load(args.model_path)
    metrics, report = evaluate(model, X, y)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{args.split_name}_metrics.json"
    report_path = args.output_dir / f"{args.split_name}_report.txt"
    metrics["task"] = args.task
    metrics_path.write_text(json.dumps(metrics, indent=2))
    report_path.write_text(report)

    print(json.dumps({"split": args.split_name, **metrics}, indent=2))
    print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
