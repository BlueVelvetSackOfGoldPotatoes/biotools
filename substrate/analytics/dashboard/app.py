from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "runs"


def list_runs() -> List[str]:
    if not RUNS_DIR.exists():
        return []
    runs = [p.name for p in RUNS_DIR.iterdir() if p.is_dir()]
    runs.sort(reverse=True)
    return runs


def load_manifest(run_id: str) -> Dict:
    p = RUNS_DIR / run_id / "manifest.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_manifest_params(manifest: Dict) -> Dict:
    raw = manifest.get("params", {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}
    return {}


def infer_model_family(manifest: Dict) -> str:
    raw_family = str(manifest.get("model_family", "unknown")).strip().lower() or "unknown"
    benchmark_id = str(manifest.get("benchmark_id", "")).strip().lower()
    params = parse_manifest_params(manifest)
    model_spec = str(params.get("model", "")).strip().lower()
    variant = str(manifest.get("model_variant", "")).strip().lower()
    algo = str(params.get("algorithm", "")).strip().lower()

    if benchmark_id == "tictactoe":
        if model_spec:
            return "hybrid" if model_spec.startswith("hybrid:") else model_spec
        if variant.startswith("bit_bridge_"):
            suffix = variant[len("bit_bridge_"):]
            if suffix.endswith("_deep_q"):
                suffix = suffix[: -len("_deep_q")]
            return "hybrid" if suffix.startswith("hybrid_") else (suffix.split("_", 1)[0] or "unknown")
        if "tabular" in variant or variant.startswith("q_learning") or variant.startswith("sarsa") or algo in {"q_learning", "sarsa", "dqn", "ddqn", "muzero_lite", "deep_q"}:
            return "reinforcement"
        return "unknown"

    if raw_family == "tictactoe":
        return "reinforcement"
    return raw_family


def load_csv(run_id: str, rel: str) -> pd.DataFrame | None:
    p = RUNS_DIR / run_id / rel
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def page_overview(run_id: str):
    st.subheader("Overview")
    manifest = load_manifest(run_id)
    if not manifest:
        st.info("No manifest for selected run.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Run: `{manifest.get('run_id', run_id)}`")
        st.write(f"Model family: `{infer_model_family(manifest)}`")
        st.write(f"Variant: `{manifest.get('model_variant', 'unknown')}`")
        st.write(f"Seed: `{manifest.get('seed', 'n/a')}`")
    with col2:
        st.write(f"Data version: `{manifest.get('data_version', 'n/a')}`")
        st.write(f"Start: `{manifest.get('train_start_utc', 'n/a')}`")
        st.write(f"End: `{manifest.get('train_end_utc', 'n/a')}`")

    epoch = load_csv(run_id, "learning/epoch_metrics.csv")
    if epoch is not None and not epoch.empty:
        st.markdown("**Latest Metrics**")
        latest_epoch = int(epoch["epoch"].max()) if "epoch" in epoch.columns else None
        latest = epoch[epoch["epoch"] == latest_epoch] if latest_epoch is not None else epoch
        st.dataframe(latest, use_container_width=True)

    infer = load_csv(run_id, "deployment/inference_metrics.csv")
    if infer is not None and not infer.empty:
        avg_conf = float(infer["confidence"].mean()) if "confidence" in infer else float("nan")
        avg_lat = float(infer["latency_ms"].mean()) if "latency_ms" in infer else float("nan")
        acc = float(infer["is_correct"].mean()) if "is_correct" in infer else float("nan")
        c1, c2, c3 = st.columns(3)
        c1.metric("Deployment Accuracy", f"{acc*100:.2f}%")
        c2.metric("Mean Confidence", f"{avg_conf:.4f}")
        c3.metric("Mean Latency (ms)", f"{avg_lat:.4f}")


def page_training(run_id: str):
    st.subheader("Training")
    epoch = load_csv(run_id, "learning/epoch_metrics.csv")
    if epoch is None or epoch.empty:
        st.info("No epoch metrics available.")
        return

    numeric = [c for c in ["loss", "accuracy", "grad_norm_mean", "param_norm_mean", "samples_per_sec"] if c in epoch.columns]
    metric = st.selectbox("Metric", numeric, index=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    for split in sorted(epoch["split"].dropna().unique()):
        d = epoch[epoch["split"] == split].sort_values("epoch")
        ax.plot(d["epoch"], d[metric], marker="o", label=str(split))
    ax.set_title(f"{metric} by epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    cm = load_csv(run_id, "learning/confusion_matrix.csv")
    if cm is not None and not cm.empty:
        split = st.selectbox("Confusion split", sorted(cm["split"].unique()), key="cm_split")
        d = cm[cm["split"] == split]
        if not d.empty:
            epoch_opts = sorted(d["epoch"].unique())
            chosen_epoch = st.selectbox("Confusion epoch", epoch_opts, index=len(epoch_opts)-1)
            d = d[d["epoch"] == chosen_epoch]
            piv = d.pivot_table(index="true_class", columns="pred_class", values="count", aggfunc="sum", fill_value=0)

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(piv.values, cmap="Blues")
            ax.set_title(f"Confusion matrix ({split}, epoch={chosen_epoch})")
            ax.set_xlabel("pred")
            ax.set_ylabel("true")
            ax.set_xticks(range(len(piv.columns)))
            ax.set_yticks(range(len(piv.index)))
            ax.set_xticklabels(piv.columns)
            ax.set_yticklabels(piv.index)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)


def page_deployment(run_id: str):
    st.subheader("Deployment")
    infer = load_csv(run_id, "deployment/inference_metrics.csv")
    cal = load_csv(run_id, "deployment/calibration_bins.csv")
    sysm = load_csv(run_id, "deployment/system_metrics.csv")

    if infer is None or infer.empty:
        st.info("No inference metrics available.")
        return

    if "confidence" in infer.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(infer["confidence"].dropna(), bins=30, alpha=0.8)
        ax.set_title("Confidence distribution")
        ax.set_xlabel("confidence")
        ax.set_ylabel("count")
        st.pyplot(fig)
        plt.close(fig)

    if "latency_ms" in infer.columns:
        lat = infer["latency_ms"].dropna().to_numpy()
        if len(lat):
            p50, p95, p99 = np.percentile(lat, [50, 95, 99])
            st.write(f"Latency percentiles: P50={p50:.4f}ms, P95={p95:.4f}ms, P99={p99:.4f}ms")

    if cal is not None and not cal.empty:
        split = st.selectbox("Calibration split", sorted(cal["split"].unique()), key="cal_split")
        d = cal[cal["split"] == split].sort_values("bin_id")
        if not d.empty:
            x = (d["conf_low"] + d["conf_high"]) / 2.0
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
            ax.plot(x, d["empirical_acc"], marker="o", label="Empirical")
            ax.plot(x, d["avg_conf"], marker="x", label="Avg conf")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title("Calibration")
            ax.set_xlabel("confidence")
            ax.set_ylabel("accuracy")
            ax.legend()
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    if sysm is not None and not sysm.empty and "qps" in sysm.columns and "p95_ms" in sysm.columns:
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax2 = ax1.twinx()
        x = np.arange(len(sysm))
        ax1.plot(x, sysm["qps"], color="tab:blue", marker="o")
        ax2.plot(x, sysm["p95_ms"], color="tab:red", marker="x")
        ax1.set_title("QPS vs P95")
        ax1.set_xlabel("sample")
        ax1.set_ylabel("QPS", color="tab:blue")
        ax2.set_ylabel("P95 ms", color="tab:red")
        st.pyplot(fig)
        plt.close(fig)


def page_model_specific(run_id: str):
    st.subheader("Model-Specific")
    manifest = load_manifest(run_id)
    inferred_family = infer_model_family(manifest)
    raw_family = str(manifest.get("model_family", "")).strip()
    candidate_families = [f for f in [inferred_family, raw_family] if f]
    base = None
    family = inferred_family or raw_family or "unknown"
    for candidate in candidate_families:
        candidate_path = RUNS_DIR / run_id / "model_specific" / candidate
        if candidate_path.exists():
            base = candidate_path
            family = candidate
            break
    if base is None:
        st.info(f"No model-specific directory found for `{family}`.")
        return

    csv_files = sorted(base.glob("*.csv"))
    if not csv_files:
        st.info("No model-specific CSV files found.")
        return

    chosen = st.selectbox("CSV", [f.name for f in csv_files])
    df = pd.read_csv(base / chosen)
    st.dataframe(df.head(200), use_container_width=True)

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 2:
        x = st.selectbox("X", num_cols, index=0)
        y = st.selectbox("Y", num_cols, index=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df[x], df[y], marker="o", linewidth=1.2)
        ax.set_title(f"{chosen}: {y} vs {x}")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)


def page_compare_runs(run_ids: List[str]):
    st.subheader("Compare Runs")
    selected = st.multiselect("Runs", run_ids, default=run_ids[: min(3, len(run_ids))])
    if not selected:
        st.info("Select one or more runs.")
        return

    rows = []
    for rid in selected:
        e = load_csv(rid, "learning/epoch_metrics.csv")
        m = load_manifest(rid)
        if e is None or e.empty:
            continue
        test = e[e["split"] == "test"]
        if test.empty:
            continue
        last_epoch = int(test["epoch"].max())
        last = test[test["epoch"] == last_epoch].iloc[-1]
        rows.append({
            "run_id": rid,
            "model_family": infer_model_family(m),
            "epoch": last_epoch,
            "test_loss": float(last.get("loss", np.nan)),
            "test_accuracy": float(last.get("accuracy", np.nan)),
        })

    if not rows:
        st.info("No comparable run metrics available.")
        return

    comp = pd.DataFrame(rows)
    st.dataframe(comp, use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(comp["run_id"], comp["test_accuracy"])
    ax.set_title("Final test accuracy by run")
    ax.set_ylabel("accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def page_data_drift(run_id: str):
    st.subheader("Data / Drift")
    infer = load_csv(run_id, "deployment/inference_metrics.csv")
    if infer is None or infer.empty:
        st.info("No inference data for drift view.")
        return

    if "sample_id" not in infer.columns:
        infer = infer.reset_index().rename(columns={"index": "sample_id"})

    infer = infer.sort_values("sample_id")
    n = len(infer)
    if n < 20:
        st.info("Need more samples for drift windows.")
        return

    window = max(1, n // 10)
    chunks = []
    for i in range(0, n, window):
        d = infer.iloc[i : i + window]
        chunks.append(
            {
                "window_start": int(d["sample_id"].min()),
                "window_end": int(d["sample_id"].max()),
                "mean_confidence": float(d["confidence"].mean()) if "confidence" in d else np.nan,
                "accuracy": float(d["is_correct"].mean()) if "is_correct" in d else np.nan,
                "mean_entropy": float(d["entropy"].mean()) if "entropy" in d else np.nan,
            }
        )

    drift = pd.DataFrame(chunks)
    st.dataframe(drift, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(drift.index, drift["mean_confidence"], marker="o", label="mean_confidence")
    ax.plot(drift.index, drift["accuracy"], marker="x", label="accuracy")
    ax.set_title("Windowed confidence / accuracy")
    ax.set_xlabel("window index")
    ax.set_ylabel("value")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.set_page_config(page_title="Substrate Benchmarks Dashboard", layout="wide")
    st.title("Substrate Benchmark Analytics Dashboard")

    runs = list_runs()
    if not runs:
        st.warning("No runs found. Generate logs under runs/<run_id>/ first.")
        return

    run_id = st.sidebar.selectbox("Run", runs, index=0)
    page = st.sidebar.radio(
        "Page",
        ["Overview", "Training", "Deployment", "Model-Specific", "Compare Runs", "Data/Drift"],
        index=0,
    )

    if page == "Overview":
        page_overview(run_id)
    elif page == "Training":
        page_training(run_id)
    elif page == "Deployment":
        page_deployment(run_id)
    elif page == "Model-Specific":
        page_model_specific(run_id)
    elif page == "Compare Runs":
        page_compare_runs(runs)
    elif page == "Data/Drift":
        page_data_drift(run_id)


if __name__ == "__main__":
    main()
