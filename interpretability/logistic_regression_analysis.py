#!/usr/bin/env python
"""Interpretability report for the logistic regression churn model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="model_outputs/logistic_regression.joblib",
        type=Path,
        help="Trained logistic regression pipeline path.",
    )
    parser.add_argument(
        "--features-path",
        default="feature_outputs/features.csv",
        type=Path,
        help="Feature CSV used during training (to recover column order).",
    )
    parser.add_argument(
        "--output-dir",
        default="interpretability_outputs/logistic_regression",
        type=Path,
        help="Directory to store coefficient tables and plots.",
    )
    parser.add_argument(
        "--top-k",
        default=15,
        type=int,
        help="Number of largest coefficients (by absolute value) to visualize.",
    )
    return parser.parse_args()


def load_coefficients(model_path: Path) -> pd.Series:
    """Extract standardized coefficients from the logistic regression pipeline."""
    pipeline = joblib.load(model_path)
    clf = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    coef = pd.Series(clf.coef_[0], index=scaler.feature_names_in_)
    coef.name = "coefficient"
    return coef.sort_values(ascending=False)


def summarize_coefficients(coef: pd.Series, output_dir: Path) -> pd.DataFrame:
    """Write coefficient summary to TSV."""
    summary = pd.DataFrame(
        {
            "feature": coef.index,
            "coefficient": coef.values,
            "abs_coefficient": coef.abs().values,
        }
    ).sort_values("abs_coefficient", ascending=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "coefficients.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    return summary


def plot_coefficients(summary: pd.DataFrame, top_k: int, output_dir: Path) -> None:
    """Plot top positive and negative coefficients."""
    top = summary.head(top_k)
    plt.figure(figsize=(8, max(4, top_k * 0.3)))
    sns.barplot(
        data=top,
        y="feature",
        x="coefficient",
        hue="coefficient",
        palette="coolwarm",
        legend=False,
    )
    plt.title(f"Top {top_k} Logistic Regression Coefficients (standardized features)")
    plt.xlabel("Coefficient (impact on churn log-odds)")
    plt.ylabel("")
    plt.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "top_coefficients.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    coef = load_coefficients(args.model_path)
    summary = summarize_coefficients(coef, args.output_dir)
    plot_coefficients(summary, args.top_k, args.output_dir)
    print(
        json.dumps(
            {
                "model_path": str(args.model_path),
                "output_dir": str(args.output_dir),
                "top_k": args.top_k,
                "features_analyzed": len(summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
