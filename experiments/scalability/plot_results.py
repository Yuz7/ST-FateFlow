from __future__ import annotations

import argparse
from pathlib import Path

from common import RESULTS_ROOT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["stfateflow", "stvcr-default", "stvcr-all-cells"]
PALETTE = {
    "stfateflow": "#176B87",
    "stvcr-default": "#D1495B",
    "stvcr-all-cells": "#F28E2B",
}
LABELS = {
    "stfateflow": "ST-FateFlow",
    "stvcr-default": "stVCR-default",
    "stvcr-all-cells": "stVCR-all-cells",
}


def _existing_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    existing = [column for column in columns if column in frame]
    if not existing:
        return pd.Series(np.nan, index=frame.index)
    return frame[existing].fillna(0).sum(axis=1)


def _existing_max(frame: pd.DataFrame, suffix: str) -> pd.Series:
    columns = [column for column in frame if column.endswith(suffix)]
    if not columns:
        return pd.Series(np.nan, index=frame.index)
    return frame[columns].max(axis=1, skipna=True)


def prepare_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame["status"] == "complete"].copy()
    frame["method_label"] = frame["method"].map(LABELS)
    frame["alignment_seconds"] = _existing_sum(
        frame,
        [
            "fugwot_spatial_alignment_preprocessing__wall_seconds",
            "spatial_alignment_preprocessing__wall_seconds",
        ],
    )
    frame["preprocessing_seconds"] = _existing_sum(
        frame,
        [
            "autoencoder_preprocessing__wall_seconds",
            "fugwot_spatial_alignment_preprocessing__wall_seconds",
            "spatial_alignment_preprocessing__wall_seconds",
        ],
    )
    frame["peak_gpu_mib"] = _existing_max(frame, "__peak_process_gpu_mib")
    return frame


def _lineplot(ax, frame: pd.DataFrame, y: str, ylabel: str) -> None:
    for method in METHOD_ORDER:
        current = frame.loc[frame["method"] == method]
        if current.empty or y not in current:
            continue
        grouped = current.groupby("n_train_total", as_index=False)[y].agg(["mean", "std"]).reset_index()
        ax.plot(
            grouped["n_train_total"],
            grouped["mean"],
            marker="o",
            linewidth=1.8,
            color=PALETTE[method],
            label=LABELS[method],
        )
        if grouped["std"].notna().any():
            std = grouped["std"].fillna(0)
            ax.fill_between(
                grouped["n_train_total"],
                grouped["mean"] - std,
                grouped["mean"] + std,
                color=PALETTE[method],
                alpha=0.15,
                linewidth=0,
            )
    ax.set_xlabel("Training cells (9.5 + 11.5)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.6)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MOSTA scalability benchmark")
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULTS_ROOT / "benchmark_summary.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_summary(args.summary)
    sns.set_theme(style="white", context="paper")

    resource_metrics = [
        ("alignment_seconds", "Spatial alignment / FUGWOT time (s)"),
        ("training__wall_seconds", "Training time (s)"),
        ("total_profiled_seconds", "End-to-end profiled time (s)"),
        ("peak_gpu_mib", "Peak process GPU memory (MiB)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (metric, label) in zip(axes.flat, resource_metrics, strict=True):
        _lineplot(ax, frame, metric, label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.output_dir / "mosta_scalability_resources.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "mosta_scalability_resources.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    accuracy_metrics = [
        ("rna_mmd_rbf", "RNA MMD"),
        ("rna_wasserstein_1", "RNA Wasserstein-1"),
        ("spatial_chamfer_l2", "Spatial Chamfer distance"),
        ("spatial_hausdorff95", "Spatial HD95"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (metric, label) in zip(axes.flat, accuracy_metrics, strict=True):
        _lineplot(ax, frame, metric, label)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(args.output_dir / "mosta_scalability_accuracy.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "mosta_scalability_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
