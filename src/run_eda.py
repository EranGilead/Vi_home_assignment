#!/usr/bin/env python
"""EDA artifact generator for the Vi home assignment dataset.

This script loads the training CSV files, creates descriptive summaries, writes a
plain-text recap, and saves key visualizations that will guide downstream
feature engineering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        argparse.Namespace: Namespace with `data_dir` and `output_dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data/train",
        type=Path,
        help="Directory that holds churn_labels.csv, app_usage.csv, web_visits.csv, claims.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="eda_outputs",
        type=Path,
        help="Directory to store generated plots and text summary.",
    )
    return parser.parse_args()


def load_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the four training tables into data frames.

    Args:
        data_dir (Path): Directory containing the CSV files.

    Returns:
        dict[str, pd.DataFrame]: Mapping of logical table names to DataFrames.
    """
    tables = {
        "labels": pd.read_csv(data_dir / "churn_labels.csv", parse_dates=["signup_date"]),
        "app": pd.read_csv(data_dir / "app_usage.csv", parse_dates=["timestamp"]),
        "web": pd.read_csv(data_dir / "web_visits.csv", parse_dates=["timestamp"]),
        "claims": pd.read_csv(data_dir / "claims.csv", parse_dates=["diagnosis_date"]),
    }
    return tables


def write_summary(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write a plain-text EDA summary.

    Args:
        tables (dict[str, pd.DataFrame]): Loaded tables keyed by name.
        output_dir (Path): Destination directory for the summary text file.
    """
    labels, app, web, claims = tables["labels"], tables["app"], tables["web"], tables["claims"]
    member_count = labels["member_id"].nunique()
    churn_rate = labels["churn"].mean()
    outreach_rate = labels["outreach"].mean()
    churn_by_outreach = labels.groupby("outreach")["churn"].mean()

    app_counts = app.groupby("member_id").size()
    web_counts = web.groupby("member_id").size()
    claim_counts = claims.groupby("member_id").size()
    icd_counts = claims["icd_code"].value_counts().head(10)
    title_counts = web["title"].value_counts().head(10)

    lines = [
        "WELLCO DATASET – EDA SNAPSHOT",
        "--------------------------------",
        f"Members in train set       : {member_count:,}",
        f"Churn rate                 : {churn_rate:0.2%}",
        f"Outreach rate              : {outreach_rate:0.2%}",
        "Churn by outreach flag     : "
        + ", ".join([f"{flag} -> {rate:0.2%}" for flag, rate in churn_by_outreach.items()]),
        "",
        f"App sessions per member    : mean {app_counts.mean():0.2f}, median {app_counts.median():0.0f}",
        f"Web visits per member      : mean {web_counts.mean():0.2f}, median {web_counts.median():0.0f}",
        f"Claims per member          : mean {claim_counts.mean():0.2f}, median {claim_counts.median():0.0f}",
        "",
        "Top ICD-10 codes:",
    ]
    lines += [f"  - {code}: {count:,}" for code, count in icd_counts.items()]
    lines += ["", "Most visited content titles:"]
    lines += [f"  - {title}: {count:,}" for title, count in title_counts.items()]

    summary_path = output_dir / "eda_summary.txt"
    summary_path.write_text("\n".join(lines))


def plot_churn_outreach(labels: pd.DataFrame, output_dir: Path) -> None:
    """Plot churn rate split by outreach flag.

    Args:
        labels (pd.DataFrame): Label table with churn/outreach columns.
        output_dir (Path): Directory for the resulting PNG file.
    """
    rates = (
        labels.groupby("outreach")["churn"]
        .agg(churn_rate="mean", members="count")
        .reset_index()
    )
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=rates,
        x="outreach",
        y="churn_rate",
        hue="outreach",
        palette="viridis",
        legend=False,
    )
    plt.title("Churn Rate by Outreach Flag")
    plt.ylabel("Churn Rate")
    plt.xlabel("Outreach (0 = control, 1 = outreach)")
    for idx, row in rates.iterrows():
        plt.text(
            idx, # type: ignore
            row["churn_rate"] + 0.003,
            f"{row['churn_rate']:.1%}\n(n={row['members']})",
            ha="center",
        )
    plt.ylim(0, rates["churn_rate"].max() * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / "churn_by_outreach.png", dpi=200)
    plt.close()


def plot_engagement_histograms(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Plot per-member engagement histograms for each channel.

    Args:
        tables (dict[str, pd.DataFrame]): Loaded tables keyed by name.
        output_dir (Path): Directory for the resulting PNG file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, df) in zip(
        axes,
        [
            ("App sessions", tables["app"]),
            ("Web visits", tables["web"]),
            ("Claims", tables["claims"]),
        ],
    ):
        counts = df.groupby("member_id").size()
        sns.histplot(counts, bins=30, ax=ax) # type: ignore
        ax.set_title(f"{name} per Member")
        ax.set_xlabel("Events in 14-day window")
        ax.set_ylabel("Members")
    plt.tight_layout()
    plt.savefig(output_dir / "engagement_histograms.png", dpi=200)
    plt.close()


def plot_icd_counts(claims: pd.DataFrame, output_dir: Path) -> None:
    """Plot top ICD-10 codes by frequency.

    Args:
        claims (pd.DataFrame): Claims table containing `icd_code`.
        output_dir (Path): Directory for the resulting PNG file.
    """
    top_icd = (
        claims["icd_code"]
            .value_counts()
            .head(10)
            .rename_axis("icd_code")
            .reset_index(name="count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top_icd,
        x="count",
        y="icd_code",
        hue="icd_code",
        palette="crest",
        legend=False,
    )
    plt.title("Top 10 ICD-10 Codes")
    plt.xlabel("Claim Count")
    plt.ylabel("ICD-10 code")
    plt.tight_layout()
    plt.savefig(output_dir / "top_icd_codes.png", dpi=200)
    plt.close()


def plot_web_titles(web: pd.DataFrame, output_dir: Path) -> None:
    """Plot the most visited web content titles.

    Args:
        web (pd.DataFrame): Web visits table containing `title`.
        output_dir (Path): Directory for the resulting PNG file.
    """
    top_titles = (
        web["title"]
            .value_counts()
            .head(10)
            .rename_axis("title")
            .reset_index(name="count")
    )
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=top_titles,
        x="count",
        y="title",
        hue="title",
        palette="mako",
        legend=False,
    )
    plt.title("Top 10 Web Content Titles")
    plt.xlabel("Visit Count")
    plt.ylabel("Title")
    plt.tight_layout()
    plt.savefig(output_dir / "top_web_titles.png", dpi=200)
    plt.close()


def main() -> None:
    """Entrypoint for running the EDA workflow."""
    args = parse_args()
    tables = load_tables(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_summary(tables, args.output_dir)
    plot_churn_outreach(tables["labels"], args.output_dir)
    plot_engagement_histograms(tables, args.output_dir)
    plot_icd_counts(tables["claims"], args.output_dir)
    plot_web_titles(tables["web"], args.output_dir)

    print(f"EDA artifacts saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    sns.set_theme(style="whitegrid")
    main()
