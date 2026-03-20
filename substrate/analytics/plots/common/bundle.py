from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .common_plots import (
    plot_calibration,
    plot_confusion_matrix,
    plot_error_gallery,
    plot_latency_qps,
    plot_learning_curves,
)
from .model_specific_plots import plot_model_specific
from .io_utils import ensure_dir


def _parse_params(manifest: Dict) -> Dict:
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


def detect_model_family(run_dir: Path | str) -> str:
    run_dir = Path(run_dir)
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return "unknown"

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return "unknown"

    raw_family = str(manifest.get("model_family", "unknown")).strip().lower() or "unknown"
    benchmark_id = str(manifest.get("benchmark_id", "")).strip().lower()
    params = _parse_params(manifest)
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


def generate_common_bundle(run_dir: Path | str, out_dir: Path | str, fmt: str = "png") -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)

    outputs: List[Path] = []
    outputs += plot_learning_curves(run_dir, out_dir, fmt)
    outputs += plot_confusion_matrix(run_dir, out_dir, fmt)
    outputs += plot_calibration(run_dir, out_dir, fmt)
    outputs += plot_latency_qps(run_dir, out_dir, fmt)
    outputs += plot_error_gallery(run_dir, out_dir, fmt)
    return outputs


def generate_model_bundle(run_dir: Path | str, out_dir: Path | str, fmt: str = "png", model_family: str | None = None) -> List[Path]:
    run_dir = Path(run_dir)
    out_dir = ensure_dir(out_dir)
    family = model_family or detect_model_family(run_dir)
    return plot_model_specific(run_dir, out_dir, fmt, family)
