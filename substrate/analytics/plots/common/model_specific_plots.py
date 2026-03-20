from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from .io_utils import ensure_dir, read_csv_if_exists, save_figure


MODEL_PLOT_SPECS: Dict[str, List[dict]] = {
    "mlp": [
        {"file": "model_specific/mlp/neuron_stats.csv", "type": "line", "x": "neuron", "y": "weight_l2", "group": "epoch", "title": "MLP Neuron Weight L2"},
        {"file": "model_specific/mlp/neuron_stats.csv", "type": "line", "x": "neuron", "y": "weight_abs_mean", "group": "epoch", "title": "MLP Neuron Mean |W|"},
    ],
    "cnn": [
        {"file": "model_specific/cnn/filter_stats.csv", "type": "line", "x": "epoch", "y": "l2", "group": "layer", "title": "CNN Filter L2 Over Epochs"},
        {"file": "model_specific/cnn/featuremap_stats.csv", "type": "line", "x": "epoch", "y": "sparsity", "group": "layer", "title": "CNN Featuremap Sparsity"},
    ],
    "rnn": [
        {"file": "model_specific/rnn/timestep_stats.csv", "type": "line", "x": "timestep", "y": "grad_norm", "group": "epoch", "title": "RNN Gradient Flow By Timestep"},
        {"file": "model_specific/rnn/timestep_stats.csv", "type": "line", "x": "timestep", "y": "hidden_norm", "group": "epoch", "title": "RNN Hidden Norm By Timestep"},
    ],
    "lstm": [
        {"file": "model_specific/lstm/gate_stats.csv", "type": "line", "x": "timestep", "y": "sat_frac", "group": "gate", "title": "LSTM Gate Saturation"},
        {"file": "model_specific/lstm/gate_stats.csv", "type": "line", "x": "epoch", "y": "mean", "group": "gate", "title": "LSTM Gate Mean Over Epochs"},
    ],
    "transformer": [
        {"file": "model_specific/transformer/attention_entropy.csv", "type": "line", "x": "epoch", "y": "entropy_mean", "group": "head", "title": "Transformer Attention Entropy"},
        {"file": "model_specific/transformer/attn_weights_sampled.csv", "type": "heatmap", "index": "q", "columns": "k", "values": "attn", "title": "Transformer Attention (Sampled)"},
    ],
    "vit": [
        {"file": "model_specific/vit/cls_embedding_stats.csv", "type": "line", "x": "epoch", "y": "mean_norm", "title": "ViT CLS Norm"},
        {"file": "model_specific/vit/patch_attention.csv", "type": "line", "x": "patch_idx", "y": "attn", "group": "head", "title": "ViT Patch Attention"},
    ],
    "trees": [
        {"file": "model_specific/trees/tree_growth.csv", "type": "line", "x": "depth", "y": "nodes", "title": "Tree Growth"},
        {"file": "model_specific/trees/split_importance.csv", "type": "bar", "x": "feature", "y": "gain_total", "title": "Tree Split Importance"},
    ],
    "forests": [
        {"file": "model_specific/forests/oob_curve.csv", "type": "line", "x": "n_trees", "y": "oob_acc", "title": "Forest OOB Accuracy"},
        {"file": "model_specific/forests/tree_agreement.csv", "type": "line", "x": "sample", "y": "agreement_ratio", "title": "Forest Agreement"},
    ],
    "clustering": [
        {"file": "model_specific/clustering/cluster_metrics.csv", "type": "line", "x": "step", "y": "inertia", "title": "Clustering Inertia"},
        {"file": "model_specific/clustering/centroid_shift.csv", "type": "line", "x": "step", "y": "shift_l2", "group": "cluster_id", "title": "Centroid Shift"},
    ],
    "hebbian": [
        {"file": "model_specific/hebbian/hebb_update.csv", "type": "line", "x": "step", "y": "delta_w_l2", "group": "layer", "title": "Hebbian Update Magnitude"},
        {"file": "model_specific/hebbian/energy_curve.csv", "type": "line", "x": "step", "y": "energy", "title": "Hopfield Energy"},
    ],
    "actor_critic": [
        {"file": "model_specific/actor_critic/rl_train.csv", "type": "line", "x": "step", "y": "reward_mean", "title": "RL Reward"},
        {"file": "model_specific/actor_critic/rl_train.csv", "type": "line", "x": "step", "y": "entropy", "title": "Policy Entropy"},
    ],
    "reinforcement": [
        {"file": "model_specific/reinforcement/rl_algorithms.csv", "type": "line", "x": "step", "y": "test_acc", "group": "algorithm", "title": "RL Test Accuracy"},
        {"file": "model_specific/reinforcement/rl_algorithms.csv", "type": "line", "x": "step", "y": "loss", "group": "algorithm", "title": "RL Loss"},
        {"file": "model_specific/reinforcement/muzero_search.csv", "type": "line", "x": "step", "y": "visit_max", "title": "MuZero Visit Concentration"},
    ],
    "diffusion": [
        {"file": "model_specific/diffusion/timestep_loss.csv", "type": "line", "x": "timestep", "y": "loss", "group": "step_or_epoch", "title": "Diffusion Loss By Timestep"},
        {"file": "model_specific/diffusion/sample_stats.csv", "type": "line", "x": "step", "y": "std", "title": "Diffusion Sample Std"},
    ],
    "gnn": [
        {"file": "model_specific/gnn/message_norms.csv", "type": "line", "x": "epoch", "y": "msg_norm_mean", "group": "layer", "title": "GNN Message Norms"},
        {"file": "model_specific/gnn/embedding_2d.csv", "type": "scatter", "x": "x", "y": "y", "group": "class", "title": "GNN Embedding Projection"},
    ],
    "forward_forward": [
        {"file": "model_specific/forward_forward/goodness_stats.csv", "type": "line", "x": "epoch", "y": "margin", "group": "layer", "title": "Forward-Forward Margin"},
        {"file": "model_specific/forward_forward/goodness_stats.csv", "type": "line", "x": "epoch", "y": "pos_mean", "group": "layer", "title": "Forward-Forward Positive Goodness"},
    ],
    "continuous": [
        {"file": "model_specific/continuous/continuous_efficiency.csv", "type": "line", "x": "cycle", "y": "test_accuracy", "title": "Continuous Test Accuracy"},
        {"file": "model_specific/continuous/continuous_efficiency.csv", "type": "line", "x": "cycle", "y": "cumulative_samples", "title": "Continuous Cumulative Samples"},
        {"file": "model_specific/continuous/continuous_phase_metrics.csv", "type": "line", "x": "cycle", "y": "loss", "group": "phase", "title": "Continuous Phase Loss"},
    ],
    "hybrid": [
        {"file": "model_specific/hybrid/continuous_efficiency.csv", "type": "line", "x": "cycle", "y": "test_accuracy", "title": "Hybrid Test Accuracy"},
        {"file": "model_specific/hybrid/hybrid_expert_metrics.csv", "type": "line", "x": "cycle", "y": "fusion_strength", "group": "expert", "title": "Hybrid Fusion Strength By Expert"},
        {"file": "model_specific/hybrid/hybrid_expert_metrics.csv", "type": "line", "x": "cycle", "y": "expert_accuracy", "group": "expert", "title": "Hybrid Expert Accuracy"},
        {"file": "model_specific/hybrid/continuous_phase_metrics.csv", "type": "line", "x": "cycle", "y": "loss", "group": "phase", "title": "Hybrid Phase Loss"},
    ],
}

for _family in list(MODEL_PLOT_SPECS.keys()):
    MODEL_PLOT_SPECS[_family].extend(
        [
            {
                "file": f"model_specific/{_family}/bio_epoch_dynamics.csv",
                "type": "line",
                "x": "epoch",
                "y": "energy_used",
                "title": f"{_family.upper()} Bio Energy",
            },
            {
                "file": f"model_specific/{_family}/bio_layer_dynamics.csv",
                "type": "line",
                "x": "epoch",
                "y": "mask_density",
                "group": "layer",
                "title": f"{_family.upper()} Bio Mask Density",
            },
            {
                "file": f"model_specific/{_family}/bio_layer_dynamics.csv",
                "type": "line",
                "x": "epoch",
                "y": "mean_delay",
                "group": "layer",
                "title": f"{_family.upper()} Bio Delay",
            },
        ]
    )


def _plot_line(df: pd.DataFrame, spec: dict):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = spec["x"]
    y = spec["y"]
    group = spec.get("group")

    if group and group in df.columns:
        for g, d in df.groupby(group):
            d = d.sort_values(x)
            ax.plot(d[x], d[y], marker="o", linewidth=1.3, label=str(g))
        if len(ax.lines):
            ax.legend()
    else:
        d = df.sort_values(x)
        ax.plot(d[x], d[y], marker="o", linewidth=1.5)

    ax.set_title(spec.get("title", y))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    return fig


def _plot_bar(df: pd.DataFrame, spec: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = spec["x"]
    y = spec["y"]
    d = df.sort_values(y, ascending=False).head(20)
    ax.bar(d[x].astype(str), d[y])
    ax.set_title(spec.get("title", y))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    return fig


def _plot_heatmap(df: pd.DataFrame, spec: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    pivot = df.pivot_table(index=spec["index"], columns=spec["columns"], values=spec["values"], aggfunc="mean", fill_value=0)
    im = ax.imshow(pivot.values, cmap="viridis")
    ax.set_title(spec.get("title", "heatmap"))
    ax.set_xlabel(spec["columns"])
    ax.set_ylabel(spec["index"])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


def _plot_scatter(df: pd.DataFrame, spec: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    x = spec["x"]
    y = spec["y"]
    group = spec.get("group")
    if group and group in df.columns:
        for g, d in df.groupby(group):
            ax.scatter(d[x], d[y], s=12, alpha=0.7, label=str(g))
        ax.legend(markerscale=1.5, fontsize=8)
    else:
        ax.scatter(df[x], df[y], s=12, alpha=0.7)
    ax.set_title(spec.get("title", "scatter"))
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.3)
    return fig


def plot_model_specific(run_dir: Path | str, out_dir: Path | str, fmt: str, model_family: str) -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    specs = MODEL_PLOT_SPECS.get(model_family, [])
    outputs: List[Path] = []

    for idx, spec in enumerate(specs):
        csv_path = run_dir / spec["file"]
        df = read_csv_if_exists(csv_path)
        if df is None or df.empty:
            continue

        needed = []
        if spec["type"] == "line":
            needed = [spec["x"], spec["y"]]
            if spec.get("group"):
                needed.append(spec["group"])
        elif spec["type"] == "bar":
            needed = [spec["x"], spec["y"]]
        elif spec["type"] == "heatmap":
            needed = [spec["index"], spec["columns"], spec["values"]]
        elif spec["type"] == "scatter":
            needed = [spec["x"], spec["y"]]
            if spec.get("group"):
                needed.append(spec["group"])

        needed = [c for c in needed if c]
        if any(c not in df.columns for c in needed):
            continue

        if spec["type"] == "line":
            fig = _plot_line(df, spec)
        elif spec["type"] == "bar":
            fig = _plot_bar(df, spec)
        elif spec["type"] == "heatmap":
            fig = _plot_heatmap(df, spec)
        elif spec["type"] == "scatter":
            fig = _plot_scatter(df, spec)
        else:
            continue

        out_name = f"{model_family}_specific_{idx+1}"
        outputs.append(save_figure(fig, out_dir / out_name, fmt))
        plt.close(fig)

    return outputs
