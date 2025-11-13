#!/usr/bin/env python
"""Feature planning helper for the Vi home assignment dataset.

This script aggregates member-level engagement, claims, and outreach features
aligned with the current modeling plan, writes them to disk, and produces a few
visualizations to validate their distributions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse CLI inputs.

    Returns:
        argparse.Namespace: Includes data_dir and output_dir paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/train",
        type=Path,
        help="Directory containing churn_labels.csv, app_usage.csv, web_visits.csv, claims.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="feature_outputs",
        type=Path,
        help="Destination directory for aggregated features and plots.",
    )
    return parser.parse_args()


def load_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load base tables from the given directory.

    Args:
        data_dir (Path): Folder holding the CSV source files.

    Returns:
        dict[str, pd.DataFrame]: Mapping of table names to DataFrames.
    """
    prefix = "test_" if data_dir.name.lower() == "test" else ""

    def file(name: str) -> Path:
        path = data_dir / f"{prefix}{name}"
        if not path.exists():
            # Fall back to unprefixed filename if prefixed file is missing.
            path = data_dir / name
        return path

    tables = {
        "labels": pd.read_csv(file("churn_labels.csv"), parse_dates=["signup_date"]),
        "app": pd.read_csv(file("app_usage.csv"), parse_dates=["timestamp"]),
        "web": pd.read_csv(file("web_visits.csv"), parse_dates=["timestamp"]),
        "claims": pd.read_csv(file("claims.csv"), parse_dates=["diagnosis_date"]),
    }
    return tables


def _observation_end(tables: Dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Compute the observation window end timestamp.

    Args:
        tables (dict[str, pd.DataFrame]): Loaded tables.

    Returns:
        pd.Timestamp: Maximum timestamp/diagnosis date across tables.
    """
    timestamps = [
        tables["app"]["timestamp"].max(),
        tables["web"]["timestamp"].max(),
        tables["claims"]["diagnosis_date"].max(),
    ]
    return max(timestamps)


def _observation_span_days(tables: Dict[str, pd.DataFrame]) -> int:
    """Compute observation window length in days."""
    starts = [
        tables["app"]["timestamp"].min(),
        tables["web"]["timestamp"].min(),
        tables["claims"]["diagnosis_date"].min(),
    ]
    end = _observation_end(tables)
    start = min(starts)
    return (end - start).days + 1


def _per_member_counts(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Group by member and count events.

    Args:
        df (pd.DataFrame): Source table with member_id.
        value_col (str): Column to count.

    Returns:
        pd.Series: Event counts indexed by member_id.
    """
    return df.groupby("member_id")[value_col].count()


def _recency(series: pd.Series, obs_end: pd.Timestamp) -> pd.Series:
    """Compute days since last event."""
    recency = (obs_end - series).dt.total_seconds() / (60 * 60 * 24)
    return recency


def _interval_std(df: pd.DataFrame, time_col: str) -> pd.Series:
    """Compute std-dev of inter-event intervals in days per member."""
    diffs = (
        df.sort_values(["member_id", time_col])
        .groupby("member_id")[time_col]
        .apply(lambda s: s.diff().dt.total_seconds() / (60 * 60 * 24))
    )
    return diffs.groupby("member_id").std().fillna(0.0)


def compute_features(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aggregate member-level features for modeling.

    Args:
        tables (dict[str, pd.DataFrame]): Loaded tables.

    Returns:
        pd.DataFrame: Feature matrix keyed by member_id.
    """
    labels = tables["labels"].set_index("member_id")
    app, web, claims = tables["app"], tables["web"], tables["claims"]
    obs_end = _observation_end(tables)
    window_days = _observation_span_days(tables)

    features = pd.DataFrame(index=labels.index)

    # Engagement metrics.
    app_counts = _per_member_counts(app, "timestamp")
    web_counts = _per_member_counts(web, "timestamp")
    claim_counts = _per_member_counts(claims, "diagnosis_date")

    features["app_sessions"] = app_counts
    features["web_visits"] = web_counts
    features["claims_total"] = claim_counts
    features.fillna(0, inplace=True)

    features["app_sessions_daily_avg"] = features["app_sessions"] / window_days
    features["web_visits_daily_avg"] = features["web_visits"] / window_days
    features["claim_frequency"] = features["claims_total"] / window_days

    # Recency.
    app_last = app.groupby("member_id")["timestamp"].max()
    web_last = web.groupby("member_id")["timestamp"].max()
    claim_last = claims.groupby("member_id")["diagnosis_date"].max()

    features["app_recency_days"] = _recency(app_last.reindex(features.index), obs_end)
    features["web_recency_days"] = _recency(web_last.reindex(features.index), obs_end)
    features["claim_recency_days"] = _recency(claim_last.reindex(features.index), obs_end)
    features[["app_recency_days", "web_recency_days", "claim_recency_days"]] = features[
        ["app_recency_days", "web_recency_days", "claim_recency_days"]
    ].fillna(window_days)  # members with no activity get max recency.

    # Variability of engagement.
    features["app_interval_std"] = _interval_std(app, "timestamp").reindex(features.index).fillna(0)
    features["web_interval_std"] = _interval_std(web, "timestamp").reindex(features.index).fillna(0)

    # Claims diversity and chronic indicators.
    icd_diversity = claims.groupby("member_id")["icd_code"].nunique()
    features["claims_icd_diversity"] = icd_diversity.reindex(features.index).fillna(0)

    chronic_codes = ["I10", "E11.9", "Z71.3"]
    for code in chronic_codes:
        indicator = claims.assign(has_code=lambda df, c=code: (df["icd_code"] == c).astype(int))
        features[f"has_{code.replace('.', '')}"] = (
            indicator.groupby("member_id")["has_code"].max().reindex(features.index).fillna(0)
        )

    features["chronic_any"] = features[[f"has_{code.replace('.', '')}" for code in chronic_codes]].max(axis=1)

    # Ratios.
    features["app_web_ratio"] = (
        features["app_sessions"] / features["web_visits"].replace(0, np.nan)
    ).fillna(0)
    features["claim_app_ratio"] = (
        features["claims_total"] / features["app_sessions"].replace(0, np.nan)
    ).fillna(0)

    # Outreach and interactions.
    features["outreach"] = labels["outreach"]
    features["churn"] = labels["churn"]
    low_app_threshold = features["app_sessions"].median()
    low_web_threshold = features["web_visits"].median()
    low_engagement = ((features["app_sessions"] <= low_app_threshold) & (features["web_visits"] <= low_web_threshold)).astype(int)
    features["low_engagement_flag"] = low_engagement
    features["outreach_x_low_engagement"] = features["outreach"] * low_engagement

    return features.reset_index().rename(columns={"index": "member_id"})


def write_feature_summary(features: pd.DataFrame, output_dir: Path) -> None:
    """Write a textual snapshot of the feature set."""
    numeric_cols = features.select_dtypes(include=[np.number]).columns.drop("member_id", errors="ignore")
    summary_stats = features[numeric_cols].describe().transpose()[["mean", "std", "min", "50%", "max"]]

    lines = ["FEATURE PLAN SNAPSHOT", "----------------------"]
    for col, row in summary_stats.iterrows():
        lines.append(
            f"{col:25s} mean={row['mean']:.2f} std={row['std']:.2f} "
            f"median={row['50%']:.2f} min={row['min']:.2f} max={row['max']:.2f}"
        )

    (output_dir / "feature_summary.txt").write_text("\n".join(lines))


def plot_feature_distributions(features: pd.DataFrame, output_dir: Path) -> None:
    """Create supporting visualizations for the engineered features."""
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=features, x="churn", y="app_sessions")
    plt.title("App Sessions vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("App sessions (14 days)")
    plt.tight_layout()
    plt.savefig(output_dir / "box_app_sessions_vs_churn.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=features, x="churn", y="claim_frequency")
    plt.title("Claim Frequency vs Churn")
    plt.xlabel("Churn")
    plt.ylabel("Claims per day")
    plt.tight_layout()
    plt.savefig(output_dir / "box_claim_frequency_vs_churn.png", dpi=200)
    plt.close()

    corr_cols = [
        "app_sessions",
        "web_visits",
        "claims_total",
        "claims_icd_diversity",
        "app_recency_days",
        "web_recency_days",
        "claim_recency_days",
        "outreach",
        "churn",
    ]
    corr = features[corr_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_correlation_heatmap.png", dpi=200)
    plt.close()


def main() -> None:
    """Entrypoint for the feature-planning workflow."""
    args = parse_args()
    tables = load_tables(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    features = compute_features(tables)
    features.to_csv(args.output_dir / "features.csv", index=False)
    write_feature_summary(features, args.output_dir)
    plot_feature_distributions(features, args.output_dir)

    print(f"Feature artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
