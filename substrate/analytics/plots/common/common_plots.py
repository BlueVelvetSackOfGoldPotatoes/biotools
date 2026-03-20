from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import ensure_dir, latest_epoch, read_csv_if_exists, save_figure


def _line_by_split(df: pd.DataFrame, metric: str, title: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    for split in sorted(df["split"].dropna().unique()):
        d = df[df["split"] == split].sort_values("epoch")
        if metric not in d.columns:
            continue
        ax.plot(d["epoch"], d[metric], marker="o", linewidth=1.5, label=str(split))
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    if len(ax.lines):
        ax.legend()
    return fig


def plot_learning_curves(run_dir: Path | str, out_dir: Path | str, fmt: str = "png") -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    outputs: List[Path] = []

    df = read_csv_if_exists(
        run_dir / "learning" / "epoch_metrics.csv",
        required_columns=["epoch", "split", "loss", "accuracy"],
    )
    if df is None or df.empty:
        return outputs

    for metric, title in [
        ("loss", "Loss By Epoch"),
        ("accuracy", "Accuracy By Epoch"),
        ("grad_norm_mean", "Gradient Norm Mean By Epoch"),
        ("param_norm_mean", "Parameter Norm Mean By Epoch"),
    ]:
        if metric not in df.columns:
            continue
        fig = _line_by_split(df, metric, title)
        outputs.append(save_figure(fig, out_dir / f"learning_{metric}", fmt))
        plt.close(fig)

    return outputs


def plot_confusion_matrix(
    run_dir: Path | str,
    out_dir: Path | str,
    fmt: str = "png",
    split: str = "test",
    epoch: Optional[int] = None,
) -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    outputs: List[Path] = []

    df = read_csv_if_exists(
        run_dir / "learning" / "confusion_matrix.csv",
        required_columns=["epoch", "split", "true_class", "pred_class", "count"],
    )
    if df is None or df.empty:
        return outputs

    d = df[df["split"] == split]
    if d.empty:
        return outputs

    if epoch is None:
        epoch = latest_epoch(d)
    d = d[d["epoch"] == epoch]
    if d.empty:
        return outputs

    pivot = d.pivot_table(index="true_class", columns="pred_class", values="count", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(pivot.values, cmap="Blues")
    ax.set_title(f"Confusion Matrix ({split}, epoch={epoch})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    outputs.append(save_figure(fig, out_dir / f"confusion_{split}_epoch_{epoch}", fmt))
    plt.close(fig)
    return outputs


def plot_calibration(run_dir: Path | str, out_dir: Path | str, fmt: str = "png", split: str = "test") -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    outputs: List[Path] = []

    bins = read_csv_if_exists(
        run_dir / "deployment" / "calibration_bins.csv",
        required_columns=["split", "conf_low", "conf_high", "avg_conf", "empirical_acc", "count"],
    )
    if bins is None or bins.empty:
        return outputs

    d = bins[bins["split"] == split].sort_values("bin_id")
    if d.empty:
        return outputs

    center = (d["conf_low"] + d["conf_high"]) / 2.0

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", label="Perfect")
    ax.plot(center, d["empirical_acc"], marker="o", label="Empirical")
    ax.plot(center, d["avg_conf"], marker="x", label="Avg confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Calibration ({split})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.3)
    ax.legend()

    outputs.append(save_figure(fig, out_dir / f"calibration_{split}", fmt))
    plt.close(fig)
    return outputs


def plot_latency_qps(run_dir: Path | str, out_dir: Path | str, fmt: str = "png") -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    outputs: List[Path] = []

    infer = read_csv_if_exists(run_dir / "deployment" / "inference_metrics.csv", required_columns=["latency_ms"])
    if infer is not None and not infer.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        vals = infer["latency_ms"].dropna().to_numpy()
        if len(vals):
            p50 = np.percentile(vals, 50)
            p95 = np.percentile(vals, 95)
            p99 = np.percentile(vals, 99)
            ax.hist(vals, bins=30, alpha=0.8)
            for p, c, n in [(p50, "green", "P50"), (p95, "orange", "P95"), (p99, "red", "P99")]:
                ax.axvline(p, linestyle="--", color=c, label=f"{n}={p:.3f}ms")
            ax.legend()
        ax.set_title("Inference Latency Distribution")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        outputs.append(save_figure(fig, out_dir / "latency_distribution", fmt))
        plt.close(fig)

    sysm = read_csv_if_exists(
        run_dir / "deployment" / "system_metrics.csv",
        required_columns=["timestamp_utc", "qps", "p50_ms", "p95_ms", "p99_ms"],
    )
    if sysm is not None and not sysm.empty:
        d = sysm.copy()
        x = np.arange(len(d))
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        ax1.plot(x, d["qps"], color="tab:blue", marker="o", label="QPS")
        ax2.plot(x, d["p95_ms"], color="tab:red", marker="x", label="P95 ms")
        ax1.set_title("QPS and P95 Latency")
        ax1.set_xlabel("Sample")
        ax1.set_ylabel("QPS", color="tab:blue")
        ax2.set_ylabel("P95 (ms)", color="tab:red")
        ax1.grid(alpha=0.3)
        outputs.append(save_figure(fig, out_dir / "qps_p95", fmt))
        plt.close(fig)

    return outputs


def plot_error_gallery(run_dir: Path | str, out_dir: Path | str, fmt: str = "png", max_pairs: int = 12) -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    outputs: List[Path] = []

    infer = read_csv_if_exists(
        run_dir / "deployment" / "inference_metrics.csv",
        required_columns=["true_class", "pred_class", "is_correct"],
    )
    if infer is None or infer.empty:
        return outputs

    errors = infer[infer["is_correct"] == 0]
    if errors.empty:
        return outputs

    grp = (
        errors.groupby(["true_class", "pred_class"]).size().sort_values(ascending=False).head(max_pairs)
    )
    labels = [f"{t}->{p}" for (t, p) in grp.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(grp)), grp.values)
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Top Misclassification Pairs")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.3)

    outputs.append(save_figure(fig, out_dir / "error_pairs", fmt))
    plt.close(fig)
    return outputs
