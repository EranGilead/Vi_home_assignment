#!/usr/bin/env python
"""Generate prioritized outreach recommendations from a trained churn model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for recommendation generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default="model_outputs/logistic_regression.joblib",
        type=Path,
        help="Path to a trained model pipeline with predict_proba (defaults to logistic regression).",
    )
    parser.add_argument(
        "--features-path",
        default="feature_outputs/features.csv",
        type=Path,
        help="Feature CSV used for scoring.",
    )
    parser.add_argument(
        "--output-dir",
        default="recommendations",
        type=Path,
        help="Directory in which to store ranked scores, top candidates, and summary.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="Number of members to recommend. Overrides --top-percent if provided.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=None,
        help="Optional fraction (0-1) if you prefer percentage-based selection.",
    )
    parser.add_argument(
        "--evaluation-counts",
        type=str,
        default="100,200,500,1000,2000,3000,5000",
        help="Comma-separated outreach sizes (counts) for summary table.",
    )
    parser.add_argument(
        "--plot-counts",
        type=str,
        default="100,200,500,1000,2000,3000,4000,5000",
        help="Comma-separated outreach sizes (counts) for lift plot.",
    )
    parser.add_argument(
        "--task",
        default="churn",
        choices=["churn", "benefit"],
        help="Match scoring features to the model's training objective.",
    )
    return parser.parse_args()


def load_features(features_path: Path) -> pd.DataFrame:
    """Load feature matrix and drop redundant columns if necessary."""
    df = pd.read_csv(features_path)
    redundant_cols = {"claims_icd_diversity"}
    existing = redundant_cols.intersection(df.columns)
    if existing:
        df = df.drop(columns=list(existing))
    return df


def determine_selection_count(total: int, top_n: Optional[int], top_percent: Optional[float]) -> int:
    """Translate CLI args into a concrete outreach size."""
    if top_n is not None:
        return min(total, max(1, top_n))
    if top_percent is None:
        top_percent = 0.05
    top_percent = min(max(top_percent, 0.0), 1.0)
    return max(1, int(np.ceil(total * top_percent)))


def parse_counts(arg: str, total: int) -> List[int]:
    """Parse comma-separated outreach sizes (absolute counts)."""
    counts = []
    for token in arg.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            raise ValueError(f"Invalid outreach size: {value}")
        counts.append(min(total, value))
    return sorted(set(counts))


def generate_summary(rank_df: pd.DataFrame, counts: List[int]) -> List[dict]:
    """Compute cumulative metrics for various outreach sizes."""
    summaries = []
    for n in counts:
        subset = rank_df.head(n)
        entry = {
            "count": n,
            "fraction": n / len(rank_df),
            "avg_predicted_churn": float(subset["pred_churn_prob"].mean()),
            "median_predicted_churn": float(subset["pred_churn_prob"].median()),
            "low_engagement_share": float(subset["low_engagement_flag"].mean())
            if "low_engagement_flag" in subset
            else None,
        }
        if "churn" in subset.columns:
            entry["actual_churn_rate"] = float(subset["churn"].mean())
        summaries.append(entry)
    return summaries


def main() -> None:
    args = parse_args()
    df = load_features(args.features_path)

    drop_cols = {"member_id", "churn"}
    if args.task == "benefit":
        drop_cols.update({"outreach", "outreach_x_low_engagement"})
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].fillna(0)
    model = joblib.load(args.model_path)
    probs = model.predict_proba(X)[:, 1]

    rank_df = df[["member_id"] + ([ "churn" ] if "churn" in df.columns else [])].copy()
    rank_df["pred_churn_prob"] = probs
    optional_cols = [col for col in ["low_engagement_flag", "outreach"] if col in df.columns]
    for col in optional_cols:
        rank_df[col] = df[col]
    rank_df = rank_df.sort_values("pred_churn_prob", ascending=False).reset_index(drop=True)
    rank_df["priority_rank"] = rank_df.index + 1

    select_n = determine_selection_count(len(rank_df), args.top_n, args.top_percent)
    top_df = rank_df.head(select_n).copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = args.output_dir / "ranked_scores.csv"
    top_path = args.output_dir / "top_candidates.csv"
    summary_path = args.output_dir / "summary.json"
    plot_path = args.output_dir / "lift_plot.png"

    rank_df.to_csv(ranked_path, index=False)
    top_df.to_csv(top_path, index=False)

    summary_counts = parse_counts(args.evaluation_counts, len(rank_df))
    summary = {
        "model_path": str(args.model_path),
        "features_path": str(args.features_path),
        "selection_count": select_n,
        "total_members": len(rank_df),
        "selection_fraction": select_n / len(rank_df),
        "avg_predicted_churn_top_n": float(top_df["pred_churn_prob"].mean()),
        "median_predicted_churn_top_n": float(top_df["pred_churn_prob"].median()),
        "low_engagement_share_top_n": float(top_df["low_engagement_flag"].mean())
        if "low_engagement_flag" in top_df
        else None,
        "evaluation_counts": generate_summary(rank_df, summary_counts),
    }
    if "churn" in top_df.columns:
        summary["actual_churn_rate_top_n"] = float(top_df["churn"].mean())
    summary_path.write_text(json.dumps(summary, indent=2))

    plot_counts = parse_counts(args.plot_counts, len(rank_df))
    plot_cumulative_metrics(rank_df, plot_counts, plot_path)

    print(json.dumps(summary, indent=2))


def plot_cumulative_metrics(rank_df: pd.DataFrame, counts: List[int], output_path: Path) -> None:
    """Plot average predicted churn vs outreach count."""
    totals = []
    for n in counts:
        subset = rank_df.head(n)
        totals.append(
            {
                "count": n,
                "fraction": n / len(rank_df),
                "avg_predicted_churn": float(subset["pred_churn_prob"].mean()),
                "median_predicted_churn": float(subset["pred_churn_prob"].median()),
                "actual_churn_rate": float(subset["churn"].mean()) if "churn" in subset.columns else None,
            }
        )
    df_plot = pd.DataFrame(totals)
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df_plot, x="count", y="avg_predicted_churn", marker="o", label="Avg predicted churn")
    if "actual_churn_rate" in df_plot and df_plot["actual_churn_rate"].notna().any():
        sns.lineplot(data=df_plot, x="count", y="actual_churn_rate", marker="s", label="Actual churn")
    plt.title("Cumulative outreach performance vs outreach size")
    plt.xlabel("Members targeted (cumulative)")
    plt.ylabel("Average churn probability (cumulative)")
    plt.ylim(0, 1)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
