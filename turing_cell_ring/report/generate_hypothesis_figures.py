#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
FIGURE_DIR = REPORT_DIR / "figures"
RESULTS_DIR = ROOT / "results" / "hypothesis_figures"
H4_DIR = RESULTS_DIR / "h4_seed_runs"
VALIDATION_DIR = pathlib.Path(os.environ.get("VALIDATION_DIR", ROOT / "results" / "final_validation"))
SUMMARY_PATH = VALIDATION_DIR / "summary.json"


PALETTE = {
    "ink": "#14213D",
    "blue": "#2458B3",
    "teal": "#0E7490",
    "orange": "#D97706",
    "amber": "#F59E0B",
    "crimson": "#B91C1C",
    "green": "#15803D",
    "slate": "#64748B",
    "light": "#E5E7EB",
    "lighter": "#F8FAFC",
    "violet": "#6D28D9",
}

TABLE2_REFERENCE = {
    "X": [7.5, 3.5, 2.5, 2.5, 3.5, 7.5],
    "Y": [0.0, 8.0, 8.0, 8.0, 8.0, 0.0],
}


plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.edgecolor": PALETTE["slate"],
    "axes.linewidth": 0.8,
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#D7DEE7",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.8,
    "figure.facecolor": "white",
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


@dataclass
class RingConfig:
    cell_count: int
    diffusion_x_special: float
    diffusion_y_special: float
    time_unit_seconds: float


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text())


def section10_path(engine: str, filename: str) -> pathlib.Path:
    return VALIDATION_DIR / "section10" / engine / filename


def ensure_dirs() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    H4_DIR.mkdir(parents=True, exist_ok=True)


def solve_equilibrium(gamma: float) -> tuple[float, float]:
    discriminant = 2500.0 + 28.0 * (57.0 + 55.0 * gamma)
    x = (-50.0 + math.sqrt(discriminant)) / 14.0
    return x, 1.0


def jacobian(x: float, y: float) -> tuple[float, float, float, float]:
    a = (-14.0 * x - 50.0 * y) / 32.0
    b = (-50.0 * x) / 32.0
    c = (14.0 * x + 50.0 * y) / 32.0
    d = (50.0 * x - 2.0 * y) / 32.0
    return a, b, c, d


def mode_growths(ring: RingConfig, gamma: float, with_diffusion: bool) -> list[float]:
    x, y = solve_equilibrium(gamma)
    a, b, c, d = jacobian(x, y)
    values: list[float] = []
    for mode in range(ring.cell_count // 2 + 1):
        s = float(mode)
        n = float(ring.cell_count)
        sin_term = math.sin(math.pi * s / n)
        diffusion_x = 4.0 * ring.diffusion_x_special * sin_term * sin_term if with_diffusion else 0.0
        diffusion_y = 4.0 * ring.diffusion_y_special * sin_term * sin_term if with_diffusion else 0.0
        coeff_b = -(a + d) + diffusion_x + diffusion_y
        coeff_c = (a - diffusion_x) * (d - diffusion_y) - b * c
        discriminant = complex(coeff_b * coeff_b - 4.0 * coeff_c, 0.0)
        root = discriminant ** 0.5
        r1 = (-coeff_b + root) / 2.0
        r2 = (-coeff_b - root) / 2.0
        values.append(max(r1.real, r2.real))
    return values


def schedule_gamma(segments: list[dict], post_value: float, t: float) -> float:
    for segment in segments:
        start = segment["startTime"]
        end = segment["endTime"]
        if start <= t <= end:
            if math.isclose(start, end):
                return segment["endValue"]
            span = end - start
            frac = (t - start) / span
            return segment["startValue"] + frac * (segment["endValue"] - segment["startValue"])
    return post_value


def start_backend_server(port: int) -> subprocess.Popen[str]:
    command = [
        str(ROOT / "build" / "turing_ring_backend"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--static-root",
        str(ROOT / "frontend" / "dist"),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    for _ in range(80):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if payload.get("status") == "ok":
                    return process
        except Exception:
            time.sleep(0.1)
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    raise RuntimeError("Failed to start backend server for figure generation.")


def fetch_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_h4_seed_runs() -> list[dict]:
    seeds = list(range(1, 9))
    cached = []
    missing = []
    for seed in seeds:
        path = H4_DIR / f"reduced_xy_quick_seed{seed}.json"
        if path.exists():
            cached.append(load_json(path))
        else:
            missing.append((seed, path))

    if missing:
        server = start_backend_server(8093)
        try:
            for seed, path in missing:
                payload = {
                    "preset": "paper-quick",
                    "engine": "reduced_xy",
                    "seed": seed,
                    "dt": 0.01,
                    "totalTime": 80.0,
                    "enableNoise": True,
                    "noiseScale": 1.0,
                    "captureStride": 20,
                    "incipientCaptureGamma": 0.0625,
                    "incipientCaptureMode": "mode234_threshold",
                    "incipientCaptureStartTime": 32.0,
                    "incipientMode234Threshold": 0.18,
                }
                result = fetch_json("http://127.0.0.1:8093/api/simulate", payload)
                path.write_text(json.dumps(result, indent=2))
                cached.append(result)
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()

    cached.sort(key=lambda item: int(item["config"]["seed"]))
    summary = {
        "figure": "H4",
        "engine": "reduced_xy",
        "presetId": "paper-quick",
        "seeds": [item["config"]["seed"] for item in cached],
        "finalDominantModes": {
            str(item["config"]["seed"]): item["metrics"]["dominantMode"] for item in cached
        },
    }
    (RESULTS_DIR / "h4_summary.json").write_text(json.dumps(summary, indent=2))
    return cached


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=PALETTE["ink"])


def save_figure(fig: plt.Figure, name: str) -> None:
    path = FIGURE_DIR / name
    fig.savefig(path)
    plt.close(fig)


def snapshot_series(snapshot: dict, name: str) -> list[float]:
    return snapshot["species"][name]


def heatmap_matrix(result: dict, species: str) -> np.ndarray:
    columns = [snapshot["species"][species] for snapshot in result["snapshotTrace"]]
    return np.array(columns, dtype=float).T


def phase_trace(result: dict, species: str = "M1") -> tuple[np.ndarray, np.ndarray]:
    times = np.array([sample["time"] for sample in result["modeTrace"]], dtype=float)
    raw = np.array(
        [sample.get("speciesPhases", {}).get(species, sample["dominantPhaseRadians"]) for sample in result["modeTrace"]],
        dtype=float,
    )
    return times, np.unwrap(raw)


def generate_h1(summary: dict) -> dict:
    result = load_json(section10_path("reduced_xy", "best_paper_first_incipient.json"))
    schedule = result["config"]["schedule"]
    segments = schedule["segments"]
    post_value = schedule["postSegmentValue"]

    times = np.linspace(0.0, result["config"]["totalTime"], 801)
    gammas = [schedule_gamma(segments, post_value, float(t)) for t in times]
    snapshots = [
        ("Initial", result["initialSnapshot"], PALETTE["slate"]),
        ("Incipient", result["incipientSnapshot"], PALETTE["amber"]),
        ("Final", result["finalSnapshot"], PALETTE["crimson"]),
    ]

    fig = plt.figure(figsize=(10.5, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.2])
    ax_schedule = fig.add_subplot(gs[0, :])
    ax_schedule.plot(times, gammas, color=PALETTE["ink"], linewidth=2.0)
    ax_schedule.axhline(0.0, color=PALETTE["slate"], linewidth=1.0, linestyle="--")
    for label, snapshot, color in snapshots:
        ax_schedule.axvline(snapshot["time"], color=color, linewidth=1.5, linestyle=":")
        ax_schedule.scatter([snapshot["time"]], [snapshot["gamma"]], color=color, s=45, zorder=3)
        ax_schedule.text(snapshot["time"], snapshot["gamma"] + 0.02, label, color=color, ha="center", va="bottom")
    ax_schedule.set_title("Quick-cooking control schedule and profile snapshots")
    ax_schedule.set_xlabel("Time (special units)")
    ax_schedule.set_ylabel(r"$\gamma$")
    style_axis(ax_schedule)

    cells = np.arange(1, 21)
    y_limits = []
    for _, snapshot, _ in snapshots:
        y_limits.extend(snapshot["y"])
    y_min = min(y_limits) - 0.05
    y_max = max(y_limits) + 0.05

    for index, (label, snapshot, color) in enumerate(snapshots):
        ax = fig.add_subplot(gs[1, index])
        ax.plot(cells, snapshot["y"], color=color, linewidth=2.4)
        ax.fill_between(cells, snapshot["y"], 1.0, color=color, alpha=0.14)
        ax.axhline(1.0, color=PALETTE["slate"], linewidth=1.0, linestyle="--")
        ax.set_title(f"{label}: t={snapshot['time']:.2f}, $\\gamma$={snapshot['gamma']:.3f}")
        ax.set_xlabel("Cell index")
        if index == 0:
            ax.set_ylabel(r"$Y$")
        ax.set_xlim(1, 20)
        ax.set_ylim(y_min, y_max)
        style_axis(ax)

    save_figure(fig, "h1_homogeneity_break.pdf")
    return {
        "file": "h1_homogeneity_break.pdf",
        "incipientTime": result["incipientSnapshot"]["time"],
        "finalTime": result["finalSnapshot"]["time"],
    }


def generate_h2(summary: dict) -> dict:
    reduced_summary = load_json(section10_path("reduced_xy", "summary.json"))
    ring_data = reduced_summary["quickBatch"]["config"]["ring"]
    ring = RingConfig(
        cell_count=ring_data["cellCount"],
        diffusion_x_special=ring_data["diffusionXSpecial"],
        diffusion_y_special=ring_data["diffusionYSpecial"],
        time_unit_seconds=ring_data["timeUnitSeconds"],
    )

    gammas = np.linspace(-0.03, 0.07, 201)
    max_with_diffusion = [max(mode_growths(ring, float(gamma), True)) for gamma in gammas]
    max_without_diffusion = [max(mode_growths(ring, float(gamma), False)) for gamma in gammas]
    peak_gamma = reduced_summary["quickBatch"]["incipientAnalysis"]["gamma"]
    modes = np.arange(0, ring.cell_count // 2 + 1)
    growth_with = np.array(mode_growths(ring, peak_gamma, True))
    growth_without = np.array(mode_growths(ring, peak_gamma, False))
    threshold_gamma = reduced_summary["linear"]["thresholdGamma"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    axes[0].plot(gammas, max_with_diffusion, color=PALETTE["blue"], linewidth=2.4, label="With diffusion")
    axes[0].plot(gammas, max_without_diffusion, color=PALETTE["crimson"], linewidth=2.0, linestyle="--", label="No diffusion")
    axes[0].axhline(0.0, color=PALETTE["slate"], linewidth=1.0)
    axes[0].axvline(threshold_gamma, color=PALETTE["orange"], linewidth=1.2, linestyle=":")
    axes[0].axvline(peak_gamma, color=PALETTE["green"], linewidth=1.2, linestyle=":")
    axes[0].set_title("Maximum linear growth rate vs control parameter")
    axes[0].set_xlabel(r"$\gamma$")
    axes[0].set_ylabel("Largest real growth rate")
    axes[0].legend(loc="lower right")
    axes[0].text(threshold_gamma, 0.01, f"threshold {threshold_gamma:.4f}", color=PALETTE["orange"], ha="left", va="bottom")
    axes[0].text(peak_gamma, 0.03, f"quick peak {peak_gamma:.4f}", color=PALETTE["green"], ha="left", va="bottom")
    style_axis(axes[0])

    axes[1].plot(modes, growth_with, color=PALETTE["blue"], linewidth=2.4, marker="o", label="With diffusion")
    axes[1].plot(modes, growth_without, color=PALETTE["crimson"], linewidth=2.0, linestyle="--", marker="s", label="No diffusion")
    axes[1].axhline(0.0, color=PALETTE["slate"], linewidth=1.0)
    axes[1].axvspan(2.6, 4.4, color=PALETTE["amber"], alpha=0.14)
    axes[1].set_title(rf"Mode-wise growth at $\gamma={peak_gamma:.4f}$")
    axes[1].set_xlabel("Mode number")
    axes[1].set_ylabel("Real growth rate")
    axes[1].legend(loc="upper right")
    style_axis(axes[1])

    save_figure(fig, "h2_diffusion_instability.pdf")
    derived = {
        "file": "h2_diffusion_instability.pdf",
        "thresholdGamma": threshold_gamma,
        "peakGamma": peak_gamma,
        "maxGrowthWithDiffusionAtPeak": float(max(growth_with)),
        "maxGrowthWithoutDiffusionAtPeak": float(max(growth_without)),
        "mode3GrowthAtPeak": float(growth_with[3]),
        "mode4GrowthAtPeak": float(growth_with[4]),
    }
    (RESULTS_DIR / "h2_diffusion_scan.json").write_text(json.dumps({
        "gammas": gammas.tolist(),
        "maxGrowthWithDiffusion": max_with_diffusion,
        "maxGrowthWithoutDiffusion": max_without_diffusion,
        "modes": modes.tolist(),
        "growthWithDiffusionAtPeak": growth_with.tolist(),
        "growthWithoutDiffusionAtPeak": growth_without.tolist(),
        **derived,
    }, indent=2))
    return derived


def generate_h3(summary: dict) -> dict:
    quick_batch = load_json(section10_path("reduced_xy", "quick_batch.json"))
    final_match = load_json(section10_path("reduced_xy", "best_paper_first_final.json"))
    analysis = quick_batch["incipientAnalysis"]
    modes = np.arange(len(analysis["modes"]))
    real_parts = np.array([mode["real"] for mode in analysis["modes"]])
    imag_parts = np.array([mode["imag"] for mode in analysis["modes"]])
    final_amplitudes = np.array(final_match["finalSnapshot"]["yModeAmplitudes"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    axes[0].bar(modes, real_parts, color=[PALETTE["blue"] if m in (3, 4) else PALETTE["light"] for m in modes], edgecolor="white")
    axes[0].plot(modes, imag_parts, color=PALETTE["crimson"], linewidth=2.0, marker="o", label="Imaginary part")
    axes[0].axhline(0.0, color=PALETTE["slate"], linewidth=1.0)
    axes[0].set_title("Linear spectrum at the quick-cooking peak")
    axes[0].set_xlabel("Mode number")
    axes[0].set_ylabel("Growth rate / frequency")
    axes[0].legend(loc="upper right")
    style_axis(axes[0])

    axes[1].bar(modes, final_amplitudes, color=[PALETTE["orange"] if m == 3 else PALETTE["amber"] if m == 4 else PALETTE["light"] for m in modes], edgecolor="white")
    axes[1].set_title("Final Fourier content of the best quick final match")
    axes[1].set_xlabel("Mode number")
    axes[1].set_ylabel(r"$Y$ mode amplitude")
    style_axis(axes[1])

    save_figure(fig, "h3_stationary_mode_selection.pdf")
    return {
        "file": "h3_stationary_mode_selection.pdf",
        "dominantMode": int(analysis["dominantMode"]),
        "mode3Growth": float(analysis["mode3Growth"]),
        "mode4Growth": float(analysis["mode4Growth"]),
        "mode3FinalAmplitude": float(final_amplitudes[3]),
        "mode4FinalAmplitude": float(final_amplitudes[4]),
    }


def generate_h4(summary: dict) -> dict:
    runs = ensure_h4_seed_runs()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharex=True)
    legend_handles = [
        Line2D([0], [0], color=PALETTE["blue"], linewidth=2.0, label="Final winner: mode 3"),
        Line2D([0], [0], color=PALETTE["orange"], linewidth=2.0, label="Final winner: mode 4"),
    ]

    winner_counts = {3: 0, 4: 0}
    endpoint_rows = []
    for run in runs:
        seed = int(run["config"]["seed"])
        trace = run["modeTrace"]
        times = np.array([point["time"] for point in trace])
        mode3 = np.array([point["amplitudes"][3] for point in trace])
        mode4 = np.array([point["amplitudes"][4] for point in trace])
        winner = int(run["metrics"]["dominantMode"])
        winner_counts[winner] = winner_counts.get(winner, 0) + 1
        color = PALETTE["blue"] if winner == 3 else PALETTE["orange"]
        alpha = 0.95 if winner == 4 else 0.72

        axes[0].plot(times, mode3, color=color, linewidth=1.8, alpha=alpha)
        axes[1].plot(times, mode4, color=color, linewidth=1.8, alpha=alpha)
        axes[0].scatter([times[-1]], [mode3[-1]], color=color, s=18)
        axes[1].scatter([times[-1]], [mode4[-1]], color=color, s=18)
        endpoint_rows.append({
            "seed": seed,
            "winner": winner,
            "finalMode3": float(mode3[-1]),
            "finalMode4": float(mode4[-1]),
        })

    for ax, mode in zip(axes, [3, 4]):
        ax.set_yscale("log")
        ax.set_ylim(1.0e-5, 2.0)
        ax.set_xlabel("Time (special units)")
        ax.set_ylabel(f"Mode-{mode} amplitude")
        ax.axvline(32.0, color=PALETTE["slate"], linewidth=1.0, linestyle=":")
        ax.text(32.0, 1.4e-5, "incipient capture window opens", rotation=90, color=PALETTE["slate"], ha="left", va="bottom")
        style_axis(ax)

    axes[0].set_title("Mode-3 growth from the same quick preset, seeds 1-8")
    axes[1].set_title("Mode-4 growth from the same quick preset, seeds 1-8")
    axes[1].legend(handles=legend_handles, loc="upper left")

    save_figure(fig, "h4_noise_seeded_divergence.pdf")
    (RESULTS_DIR / "h4_seed_endpoints.json").write_text(json.dumps(endpoint_rows, indent=2))
    return {
        "file": "h4_noise_seeded_divergence.pdf",
        "winnerCounts": winner_counts,
    }


def generate_h5(summary: dict) -> dict:
    reduced = load_json(section10_path("reduced_xy", "quick_batch.json"))
    full = load_json(section10_path("full_chemistry", "quick_batch.json"))
    batches = [("Reduced engine", reduced, PALETTE["blue"]), ("Full chemistry", full, PALETTE["teal"])]
    modes = np.arange(len(reduced["dominantModeHistogram"]))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True, sharey=True)
    for ax, (title, batch, accent) in zip(axes, batches):
        counts = np.array(batch["dominantModeHistogram"])
        colors = []
        for mode in modes:
            if mode == 3:
                colors.append(accent)
            elif mode == 4:
                colors.append(PALETTE["orange"])
            else:
                colors.append(PALETTE["light"])
        ax.bar(modes, counts, color=colors, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Dominant mode")
        ax.set_ylabel("Replicates")
        ax.set_xticks(modes)
        ax.text(
            0.03,
            0.97,
            f"mode 3 share = {batch['shareMode3']:.3f}\nmode 4 share = {batch['shareMode4']:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            color=PALETTE["ink"],
            bbox={"facecolor": "white", "edgecolor": PALETTE["light"], "boxstyle": "round,pad=0.35"},
        )
        style_axis(ax)

    save_figure(fig, "h5_quick_mode_competition.pdf")
    return {
        "file": "h5_quick_mode_competition.pdf",
        "reducedShareMode3": float(reduced["shareMode3"]),
        "reducedShareMode4": float(reduced["shareMode4"]),
        "fullShareMode3": float(full["shareMode3"]),
        "fullShareMode4": float(full["shareMode4"]),
    }


def generate_h6(summary: dict) -> dict:
    reduced_quick = load_json(section10_path("reduced_xy", "quick_batch.json"))
    reduced_slow = load_json(section10_path("reduced_xy", "slow_batch.json"))
    full_quick = load_json(section10_path("full_chemistry", "quick_batch.json"))
    full_slow = load_json(section10_path("full_chemistry", "slow_batch.json"))

    rows = [
        ("Reduced\nquick", reduced_quick, PALETTE["blue"]),
        ("Reduced\nslow", reduced_slow, PALETTE["teal"]),
        ("Full\nquick", full_quick, PALETTE["orange"]),
        ("Full\nslow", full_slow, PALETTE["green"]),
    ]
    positions = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), constrained_layout=True)

    share_width = 0.34
    axes[0].bar(
        positions - share_width / 2,
        [row[1]["shareMode3"] for row in rows],
        width=share_width,
        color=PALETTE["blue"],
        label="Mode 3 share",
        yerr=[
            [row[1]["shareMode3"] - row[1]["shareMode3CiLow"] for row in rows],
            [row[1]["shareMode3CiHigh"] - row[1]["shareMode3"] for row in rows],
        ],
        capsize=3,
    )
    axes[0].bar(
        positions + share_width / 2,
        [row[1]["shareMode4"] for row in rows],
        width=share_width,
        color=PALETTE["orange"],
        label="Mode 4 share",
        yerr=[
            [row[1]["shareMode4"] - row[1]["shareMode4CiLow"] for row in rows],
            [row[1]["shareMode4CiHigh"] - row[1]["shareMode4"] for row in rows],
        ],
        capsize=3,
    )
    axes[0].set_xticks(positions, [row[0] for row in rows])
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Share of replicates")
    axes[0].set_title("Slow cooking collapses the mode-4 competition")
    axes[0].legend(loc="upper left")
    style_axis(axes[0])

    mean_regularity = [row[1]["meanRegularityIndex"] for row in rows]
    axes[1].bar(positions, mean_regularity, color=[row[2] for row in rows], edgecolor="white")
    for idx, (_, batch, color) in enumerate(rows):
        samples = [sample["regularityIndex"] for sample in batch["samples"]]
        offsets = np.linspace(-0.14, 0.14, len(samples))
        axes[1].scatter(np.full(len(samples), positions[idx]) + offsets, samples, color=color, s=16, alpha=0.35, linewidths=0)
    axes[1].set_xticks(positions, [row[0] for row in rows])
    axes[1].set_ylabel("Regularity index")
    axes[1].set_title("Slow cooking raises pattern regularity in both engines")
    style_axis(axes[1])

    save_figure(fig, "h6_slow_cooking_regularization.pdf")
    return {
        "file": "h6_slow_cooking_regularization.pdf",
        "reducedQuickRegularity": float(reduced_quick["meanRegularityIndex"]),
        "reducedSlowRegularity": float(reduced_slow["meanRegularityIndex"]),
        "fullQuickRegularity": float(full_quick["meanRegularityIndex"]),
        "fullSlowRegularity": float(full_slow["meanRegularityIndex"]),
    }


def generate_table2(summary: dict) -> dict:
    modern = load_json(VALIDATION_DIR / "example2_table2" / "modern_stable_ring.json")
    historical = load_json(VALIDATION_DIR / "example2_table2" / "historical_stable_ring.json")
    cells = np.arange(1, len(modern["finalSnapshot"]["x"]) + 1)
    scan = summary["table2"]["scan"]

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    for ax, species in zip(axes[:2], ["X", "Y"]):
        reference = TABLE2_REFERENCE[species]
        initial = snapshot_series(modern["initialSnapshot"], species)
        current = snapshot_series(modern["finalSnapshot"], species)
        historical_series = snapshot_series(historical["finalSnapshot"], species)
        ax.plot(cells, reference, color=PALETTE["ink"], linewidth=2.2, label="Table 2 reference")
        ax.plot(cells, initial, color=PALETTE["slate"], linewidth=1.8, linestyle=":", label="Perturbed start")
        ax.plot(cells, current, color=PALETTE["blue"], linewidth=2.5, label="Modern final")
        ax.plot(cells, historical_series, color=PALETTE["orange"], linewidth=2.0, linestyle="--", label="Historical final")
        ax.set_title(f"{species} profile on the six-cell stable ring")
        ax.set_xlabel("Cell index")
        ax.set_ylabel(species)
        ax.set_xticks(cells)
        style_axis(ax)
    axes[1].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.2))

    ax_scan = axes[2]
    ax_scan.axhline(0.0, color=PALETTE["slate"], linewidth=1.0, linestyle="--")
    ks = np.array([entry["k"] for entry in scan], dtype=float)
    growths = np.array([entry["dominantGrowth"] for entry in scan], dtype=float)
    classes = [entry["dominantClassification"] for entry in scan]
    for classification, color, marker in [
        ("stationary", PALETTE["blue"], "o"),
        ("oscillatory", PALETTE["violet"], "s"),
    ]:
        mask = [value == classification for value in classes]
        if any(mask):
            ax_scan.scatter(ks[mask], growths[mask], color=color, marker=marker, s=60, label=classification.title())
    ax_scan.plot(ks, growths, color=PALETTE["light"], linewidth=1.6, zorder=0)
    for entry in scan:
        ax_scan.annotate(
            f"f={entry['f']:.0f}",
            (entry["k"], entry["dominantGrowth"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color=PALETTE["ink"],
        )
    ax_scan.set_title("Example 2 scan across $k = 16 - f$")
    ax_scan.set_xlabel("$k$")
    ax_scan.set_ylabel("Dominant real growth")
    ax_scan.legend(loc="upper left")
    style_axis(ax_scan)

    save_figure(fig, "table2_stable_ring.pdf")
    return {
        "file": "table2_stable_ring.pdf",
        "modernInitialMaxAbsErrorX": float(summary["table2"]["referenceInitialMaxAbsError"]["modernX"]),
        "modernInitialMaxAbsErrorY": float(summary["table2"]["referenceInitialMaxAbsError"]["modernY"]),
        "modernMaxAbsErrorX": float(summary["table2"]["referenceMaxAbsError"]["modernX"]),
        "modernMaxAbsErrorY": float(summary["table2"]["referenceMaxAbsError"]["modernY"]),
        "historicalMaxAbsErrorX": float(summary["table2"]["referenceMaxAbsError"]["historicalX"]),
        "historicalMaxAbsErrorY": float(summary["table2"]["referenceMaxAbsError"]["historicalY"]),
    }


def generate_oscillatory(summary: dict) -> dict:
    case_e = load_json(VALIDATION_DIR / "oscillatory_toolkit" / "case_e_simulation.json")
    case_f = load_json(VALIDATION_DIR / "oscillatory_toolkit" / "case_f_simulation.json")
    matrix_e = heatmap_matrix(case_e, "M1")
    matrix_f = heatmap_matrix(case_f, "M1")

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.0), constrained_layout=True)
    for ax, matrix, title, result in [
        (axes[0, 0], matrix_e, "Case (e): travelling-wave toolkit", case_e),
        (axes[0, 1], matrix_f, "Case (f): short-wave oscillation", case_f),
    ]:
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0.0, result["finalSnapshot"]["time"], 1, matrix.shape[0]],
        )
        ax.set_xlabel("Time (special units)")
        ax.set_ylabel("Cell index")
        ax.set_title(
            f"{title}\n"
            f"mode {result['metrics']['dominantMode']}, "
            f"freq {result['metrics']['dominantFrequencyCyclesPerTime']:.3f}, "
            f"phase {result['metrics']['neighborPhaseOffsetRadians']:.3f}"
        )
        style_axis(ax)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    for ax, title, result, color in [
        (axes[1, 0], "Case (e): unwrapped phase drift of $M_1$", case_e, PALETTE["blue"]),
        (axes[1, 1], "Case (f): unwrapped phase drift of $M_1$", case_f, PALETTE["violet"]),
    ]:
        times, phases = phase_trace(result)
        ax.plot(times, phases, color=color, linewidth=2.2)
        ax.set_title(title)
        ax.set_xlabel("Time (special units)")
        ax.set_ylabel("Phase (radians)")
        style_axis(ax)

    save_figure(fig, "oscillatory_cases.pdf")
    return {
        "file": "oscillatory_cases.pdf",
        "caseEDominantMode": int(case_e["metrics"]["dominantMode"]),
        "caseFDominantMode": int(case_f["metrics"]["dominantMode"]),
    }


def generate_historical(summary: dict) -> dict:
    modern = load_json(VALIDATION_DIR / "historical_comparisons" / "quick_modern.json")
    historical_profile_id = summary["historicalComparisons"]["executionProfileId"]
    historical = load_json(VALIDATION_DIR / "historical_comparisons" / f"quick_{historical_profile_id}.json")
    profiles = summary["historicalComparisons"]["quickHistoricalProfiles"]
    cells = np.arange(1, len(modern["finalSnapshot"]["x"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3), constrained_layout=True)
    for ax, species in zip(axes[:2], ["X", "Y"]):
        modern_values = snapshot_series(modern["finalSnapshot"], species)
        historical_values = snapshot_series(historical["finalSnapshot"], species)
        ax.plot(cells, modern_values, color=PALETTE["blue"], linewidth=2.5, label="Modern")
        ax.plot(cells, historical_values, color=PALETTE["orange"], linewidth=2.2, linestyle="--", label="Historical")
        ax.set_title(f"Quick-cooking final {species} profile")
        ax.set_xlabel("Cell index")
        ax.set_ylabel(species)
        style_axis(ax)
    axes[1].legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))

    ax_profiles = axes[2]
    labels = [entry["executionProfileId"].replace("historic_1952_", "") for entry in profiles]
    y_deltas = [entry["finalDelta"]["y"] for entry in profiles]
    regularities = [entry["regularityIndex"] for entry in profiles]
    bars = ax_profiles.bar(labels, y_deltas, color=[PALETTE["orange"], PALETTE["amber"], PALETTE["green"], PALETTE["violet"]])
    for bar, entry, regularity in zip(bars, profiles, regularities):
        ax_profiles.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.04,
            f"m{entry['dominantMode']}\nR={regularity:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=PALETTE["ink"],
        )
    ax_profiles.set_title("Historical-profile sensitivity versus modern")
    ax_profiles.set_ylabel("Max abs final $Y$ delta")
    ax_profiles.tick_params(axis="x", rotation=18)
    style_axis(ax_profiles)

    save_figure(fig, "historical_quick_comparison.pdf")
    return {
        "file": "historical_quick_comparison.pdf",
        "historicalProfileId": historical_profile_id,
        "quickDeltaX": float(summary["historicalComparisons"]["quickFinalDelta"]["x"]),
        "quickDeltaY": float(summary["historicalComparisons"]["quickFinalDelta"]["y"]),
    }


def main() -> int:
    ensure_dirs()
    summary = load_json(SUMMARY_PATH)
    try:
        source_summary = str(SUMMARY_PATH.relative_to(ROOT))
    except ValueError:
        source_summary = str(SUMMARY_PATH)
    figure_summary = {
        "generatedAt": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sourceSummary": source_summary,
        "figures": {
            "H1": generate_h1(summary),
            "H2": generate_h2(summary),
            "H3": generate_h3(summary),
            "H4": generate_h4(summary),
            "H5": generate_h5(summary),
            "H6": generate_h6(summary),
            "Table2": generate_table2(summary),
            "Oscillatory": generate_oscillatory(summary),
            "Historical": generate_historical(summary),
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(figure_summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
