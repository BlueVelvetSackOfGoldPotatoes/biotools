#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Dict, List, Set, Tuple

from random_search import MODEL_SPECS, score_run_dir


FEATURES: List[Tuple[str, str]] = [
    ("s", "BIO_ENABLE_STRUCTURAL_PLASTICITY"),
    ("h", "BIO_ENABLE_HOMEOSTASIS"),
    ("t", "BIO_ENABLE_THREE_FACTOR"),
    ("m", "BIO_ENABLE_MYELINATION"),
    ("b", "BIO_ENABLE_BIOELECTRIC"),
    ("e", "BIO_ENABLE_EI_SIGN_CONSTRAINTS"),
]

SEED_ENV: Dict[str, str] = {
    "mlp": "MLP_SEED",
    "cnn": "CNN_SEED",
    "rnn": "RNN_SEED",
    "lstm": "LSTM_SEED",
    "transformer": "TRANSFORMER_SEED",
    "vit": "VIT_SEED",
    "hebbian": "HEBBIAN_SEED",
    "forward_forward": "FF_SEED",
    "gnn": "GNN_SEED",
    "diffusion": "DIFFUSION_SEED",
    "actor_critic": "ACTOR_CRITIC_SEED",
    "reinforcement": "RL_SEED",
    "markov": "MARKOV_SEED",
    "tictactoe": "TICTACTOE_SEED",
    "trees": "TREE_SEED",
    "forests": "FORESTS_SEED",
    "clustering": "CLUSTERING_SEED",
    "continuous": "ONLINE_SEED",
}

PROFILE_QUICK: Dict[str, str] = {
    "MLP_EPOCHS": "3",
    "CNN_EPOCHS": "2",
    "RNN_EPOCHS": "3",
    "LSTM_EPOCHS": "3",
    "TRANSFORMER_EPOCHS": "2",
    "VIT_EPOCHS": "2",
    "HEBBIAN_PASSES": "4",
    "HEBBIAN_PROBE_EPOCHS": "20",
    "FF_EPOCHS": "10",
    "GNN_EPOCHS": "8",
    "DIFFUSION_STEPS": "400",
    "ACTOR_CRITIC_STEPS": "1500",
    "ACTOR_CRITIC_EVAL_EVERY": "500",
    "RL_TRAIN_N": "2000",
    "RL_TEST_N": "400",
    "RL_Q_STEPS": "1500",
    "RL_SARSA_STEPS": "1500",
    "RL_DQN_STEPS": "1000",
    "RL_MUZERO_STEPS": "800",
    "MARKOV_EPOCHS": "4",
    "MARKOV_TRAIN_N": "12000",
    "MARKOV_TEST_N": "2500",
    "TICTACTOE_EPISODES": "1500",
    "TICTACTOE_EVAL_EVERY": "300",
    "TICTACTOE_EVAL_EPISODES": "150",
    "FORESTS_N_TREES": "30",
    "TREE_DEPTHS": "12",
    "CLUSTERING_N": "1500",
    "CLUSTERING_AGG_N": "300",
    "ONLINE_CYCLES": "3",
    "ONLINE_ACTIVE_STEPS": "8",
    "ONLINE_SLEEP_STEPS": "4",
    "ONLINE_EVAL_SUBSET": "1000",
}

PROFILE_BALANCED: Dict[str, str] = {
    "MLP_EPOCHS": "6",
    "CNN_EPOCHS": "4",
    "RNN_EPOCHS": "6",
    "LSTM_EPOCHS": "6",
    "TRANSFORMER_EPOCHS": "4",
    "VIT_EPOCHS": "4",
    "HEBBIAN_PASSES": "8",
    "HEBBIAN_PROBE_EPOCHS": "40",
    "FF_EPOCHS": "20",
    "GNN_EPOCHS": "12",
    "DIFFUSION_STEPS": "1200",
    "ACTOR_CRITIC_STEPS": "4000",
    "ACTOR_CRITIC_EVAL_EVERY": "1000",
    "RL_TRAIN_N": "5000",
    "RL_TEST_N": "800",
    "RL_Q_STEPS": "3000",
    "RL_SARSA_STEPS": "3000",
    "RL_DQN_STEPS": "2000",
    "RL_MUZERO_STEPS": "1500",
    "MARKOV_EPOCHS": "6",
    "MARKOV_TRAIN_N": "20000",
    "MARKOV_TEST_N": "4000",
    "TICTACTOE_EPISODES": "4000",
    "TICTACTOE_EVAL_EVERY": "600",
    "TICTACTOE_EVAL_EPISODES": "300",
    "FORESTS_N_TREES": "50",
    "TREE_DEPTHS": "20",
    "CLUSTERING_N": "2500",
    "CLUSTERING_AGG_N": "500",
    "ONLINE_CYCLES": "6",
    "ONLINE_ACTIVE_STEPS": "12",
    "ONLINE_SLEEP_STEPS": "6",
    "ONLINE_EVAL_SUBSET": "1500",
}


@dataclass(frozen=True)
class BioCombo:
    combo_id: str
    active: int
    bits: str
    env: Dict[str, str]


@dataclass(frozen=True)
class Task:
    task_index: int
    model: str
    combo_id: str
    repeat: int
    env: Dict[str, str]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_extra_env(extra: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not extra.strip():
        return out
    for item in extra.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --extra-env entry '{item}', expected KEY=VALUE")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def env_profile(name: str) -> Dict[str, str]:
    key = name.strip().lower()
    if key == "full":
        return {}
    if key == "quick":
        return dict(PROFILE_QUICK)
    if key == "balanced":
        return dict(PROFILE_BALANCED)
    raise ValueError(f"Unknown profile '{name}'")


def default_runtime_env() -> Dict[str, str]:
    nproc = os.cpu_count() or 4
    return {
        "OMP_NUM_THREADS": str(nproc),
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "cores",
        "TENSOR_USE_CUDA": "1",
        "BATCH_LOG_EVERY": "5",
    }


def combo_from_bits(bits: Tuple[int, ...]) -> BioCombo:
    env = {"BIO_ACTIVE": "1"}
    bit_parts: List[str] = []
    for (short, env_name), bit in zip(FEATURES, bits):
        v = "1" if bit else "0"
        env[env_name] = v
        bit_parts.append(f"{short}{v}")
    bits_str = "".join(str(b) for b in bits)
    return BioCombo(
        combo_id="bio_on_" + "_".join(bit_parts),
        active=1,
        bits=bits_str,
        env=env,
    )


def generate_full_combos(include_baseline: bool) -> List[BioCombo]:
    out: List[BioCombo] = []
    if include_baseline:
        out.append(BioCombo(combo_id="bio_observe_baseline", active=0, bits="-" * len(FEATURES), env={"BIO_ACTIVE": "0"}))
    for bits in product([0, 1], repeat=len(FEATURES)):
        out.append(combo_from_bits(tuple(int(b) for b in bits)))
    return out


def build_pair_universe() -> Set[Tuple[int, int, int, int]]:
    universe: Set[Tuple[int, int, int, int]] = set()
    idxs = list(range(len(FEATURES)))
    for i, j in combinations(idxs, 2):
        for vi in (0, 1):
            for vj in (0, 1):
                universe.add((i, vi, j, vj))
    return universe


def covered_pairs(bits: Tuple[int, ...]) -> Set[Tuple[int, int, int, int]]:
    c: Set[Tuple[int, int, int, int]] = set()
    idxs = list(range(len(FEATURES)))
    for i, j in combinations(idxs, 2):
        c.add((i, bits[i], j, bits[j]))
    return c


def generate_pairwise_combos(include_baseline: bool) -> List[BioCombo]:
    candidates = [tuple(int(b) for b in bits) for bits in product([0, 1], repeat=len(FEATURES))]
    universe = build_pair_universe()
    uncovered = set(universe)
    selected: List[Tuple[int, ...]] = []

    while uncovered:
        best_bits = None
        best_cover = -1
        for bits in candidates:
            cover = len(covered_pairs(bits) & uncovered)
            if cover > best_cover:
                best_cover = cover
                best_bits = bits
        if best_bits is None or best_cover <= 0:
            break
        selected.append(best_bits)
        uncovered -= covered_pairs(best_bits)
        candidates.remove(best_bits)

    out: List[BioCombo] = []
    if include_baseline:
        out.append(BioCombo(combo_id="bio_observe_baseline", active=0, bits="-" * len(FEATURES), env={"BIO_ACTIVE": "0"}))
    out.extend(combo_from_bits(bits) for bits in selected)
    return out


def generate_core_combos(include_baseline: bool) -> List[BioCombo]:
    out: List[BioCombo] = []
    if include_baseline:
        out.append(BioCombo(combo_id="bio_observe_baseline", active=0, bits="-" * len(FEATURES), env={"BIO_ACTIVE": "0"}))

    zeros = tuple(0 for _ in FEATURES)
    ones = tuple(1 for _ in FEATURES)
    out.append(combo_from_bits(zeros))
    for i in range(len(FEATURES)):
        bits = [0] * len(FEATURES)
        bits[i] = 1
        out.append(combo_from_bits(tuple(bits)))
    out.append(combo_from_bits(ones))
    return out


def order_combos(combos: List[BioCombo], combo_order: str) -> List[BioCombo]:
    key = combo_order.strip().lower()
    if key == "default":
        return combos
    if key != "ablation":
        raise ValueError(f"Unknown combo order '{combo_order}'")

    observe_baseline = [c for c in combos if c.active == 0]
    active = [c for c in combos if c.active == 1]
    active.sort(
        key=lambda c: (
            -sum(1 for ch in c.bits if ch == "1"),
            c.bits,
            c.combo_id,
        )
    )
    # Keep BIO_ACTIVE=0 baseline last so the sweep starts from all bio features on.
    return active + observe_baseline


def generate_combos(
    mode: str,
    include_baseline: bool,
    max_combos: int,
    combo_order: str,
) -> List[BioCombo]:
    key = mode.strip().lower()
    if key == "full":
        combos = generate_full_combos(include_baseline)
    elif key == "pairwise":
        combos = generate_pairwise_combos(include_baseline)
    elif key == "core":
        combos = generate_core_combos(include_baseline)
    else:
        raise ValueError(f"Unknown combo mode '{mode}'")

    combos = order_combos(combos, combo_order)

    if max_combos > 0:
        keep = max(1, max_combos)
        combos = combos[:keep]
    return combos


def parse_models(models_arg: str) -> List[str]:
    if models_arg.strip().lower() == "all":
        return list(MODEL_SPECS.keys())
    model_keys = [m.strip().lower() for m in models_arg.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(MODEL_SPECS.keys())}")
    return model_keys


def discover_new_runs(runs_dir: Path, before: Set[str]) -> List[str]:
    after = set(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else set()
    return sorted(after - before)


def runs_for_trial_uuid(runs_dir: Path, candidate_runs: List[str], trial_uuid: str) -> List[str]:
    if not trial_uuid:
        return candidate_runs
    matched: List[str] = []
    for run_id in candidate_runs:
        manifest_path = runs_dir / run_id / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        if str(manifest.get("trial_uuid") or "") == trial_uuid:
            matched.append(run_id)
    return matched


def run_task(
    repo: Path,
    runs_dir: Path,
    task: Task,
    model_timeout: int,
) -> Tuple[int, float, List[str], float, str]:
    spec = MODEL_SPECS[task.model]
    cmd = [str(repo / spec.binary)]
    proc_env = os.environ.copy()
    proc_env.update(task.env)

    before = set(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else set()
    start = time.time()
    rc = 0
    tail = ""
    try:
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as out:
            proc = subprocess.run(
                cmd,
                cwd=str(repo),
                env=proc_env,
                stdout=out,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=model_timeout,
            )
            rc = proc.returncode

            out.seek(0)
            tail_buf = deque(maxlen=10)
            for line in out:
                tail_buf.append(line.rstrip("\n"))
            tail = "\n".join(tail_buf)
    except subprocess.TimeoutExpired:
        rc = 124
        tail = "timeout"

    elapsed = time.time() - start
    new_runs = discover_new_runs(runs_dir, before)
    owned_runs = runs_for_trial_uuid(runs_dir, new_runs, task.env.get("TRIAL_UUID", ""))
    if not owned_runs and new_runs:
        # Backward-compatible fallback if a benchmark does not yet emit trial_uuid.
        owned_runs = new_runs
    score = float("-inf")
    for run_id in owned_runs:
        score = max(score, score_run_dir(runs_dir / run_id))
    return rc, elapsed, owned_runs, score, tail


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Queue sequential model runs across bio-feature combinations."
    )
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--models", type=str, default="all", help="comma-separated model keys or 'all'")
    ap.add_argument("--combo-mode", type=str, default="full", choices=["full", "pairwise", "core"])
    ap.add_argument(
        "--combo-order",
        type=str,
        default="default",
        choices=["default", "ablation"],
        help="Ordering for the selected combo set; 'ablation' starts from all features enabled",
    )
    ap.add_argument("--include-baseline", action="store_true", default=True)
    ap.add_argument("--no-include-baseline", action="store_true")
    ap.add_argument("--max-combos", type=int, default=0, help="truncate combo list if >0")
    ap.add_argument("--repeats", type=int, default=1, help="runs per model+combo")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--seed-mode",
        type=str,
        default="paired",
        choices=["paired", "task"],
        help="Seed assignment strategy: paired keeps model+repeat seeds constant across combos",
    )
    ap.add_argument("--profile", type=str, default="quick", choices=["quick", "balanced", "full"])
    ap.add_argument("--extra-env", type=str, default="", help="comma-separated KEY=VALUE env pairs")
    ap.add_argument("--timeout-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--build", action="store_true", default=False)
    ap.add_argument("--resume", action="store_true", default=False)
    ap.add_argument("--dry-run", action="store_true", default=False)
    args = ap.parse_args()

    repo = args.repo.resolve()
    runs_dir = repo / "runs"
    include_baseline = args.include_baseline and not args.no_include_baseline
    model_keys = parse_models(args.models)
    combos = generate_combos(args.combo_mode, include_baseline, args.max_combos, args.combo_order)
    repeats = max(1, args.repeats)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (repo / "reports" / f"bio_sweep_{ts}")
    ensure_dir(out_dir)

    plan_csv = out_dir / "queue_plan.csv"
    summary_csv = out_dir / "summary.csv"
    meta_json = out_dir / "meta.json"

    base_env = default_runtime_env()
    base_env.update(env_profile(args.profile))
    base_env.update(parse_extra_env(args.extra_env))

    tasks: List[Task] = []
    idx = 0
    for model in model_keys:
        for combo in combos:
            for rep in range(repeats):
                idx += 1
                env = dict(base_env)
                env.update(combo.env)
                env["TRIAL_UUID"] = f"bio_{model}_{combo.combo_id}_r{rep+1}_{uuid.uuid4().hex[:10]}"
                env.setdefault("JOB_ORIGIN", "bio_sweep_queue")
                seed_key = SEED_ENV.get(model)
                if seed_key:
                    if args.seed_mode == "paired":
                        env[seed_key] = str(args.seed + rep)
                    else:
                        env[seed_key] = str(args.seed + idx * 9973 + rep)
                tasks.append(
                    Task(
                        task_index=idx,
                        model=model,
                        combo_id=combo.combo_id,
                        repeat=rep + 1,
                        env=env,
                    )
                )

    with plan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task_index", "model", "combo_id", "repeat", "binary", "timeout_sec"],
        )
        writer.writeheader()
        for task in tasks:
            spec = MODEL_SPECS[task.model]
            writer.writerow(
                {
                    "task_index": task.task_index,
                    "model": task.model,
                    "combo_id": task.combo_id,
                    "repeat": task.repeat,
                    "binary": spec.binary,
                    "timeout_sec": int(max(1, math.ceil(spec.timeout_sec * args.timeout_scale))),
                }
            )

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo": str(repo),
        "models": model_keys,
        "combo_mode": args.combo_mode,
        "combo_order": args.combo_order,
        "combos": [
            {"combo_id": c.combo_id, "active": c.active, "bits": c.bits, "env": c.env} for c in combos
        ],
        "repeats": repeats,
        "profile": args.profile,
        "seed_mode": args.seed_mode,
        "include_baseline": include_baseline,
        "max_combos": args.max_combos,
        "extra_env": parse_extra_env(args.extra_env),
        "task_count": len(tasks),
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    if args.dry_run:
        print(f"Planned {len(tasks)} tasks.")
        print(f"Plan: {plan_csv}")
        print(f"Meta: {meta_json}")
        return

    if args.build:
        nj = os.cpu_count() or 4
        print(f"[build] make benchmarks -j{nj}")
        subprocess.run(["make", "benchmarks", f"-j{nj}"], cwd=str(repo), check=True)

    completed_keys: Set[Tuple[str, str, int]] = set()
    if args.resume and summary_csv.exists():
        with summary_csv.open("r", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    completed_keys.add(
                        (row["model"], row["combo_id"], int(row.get("repeat", "1")))
                    )
                except Exception:
                    continue

    write_header = not summary_csv.exists() or summary_csv.stat().st_size == 0
    with summary_csv.open("a", newline="") as f:
        fieldnames = [
            "task_index",
            "model",
            "combo_id",
            "repeat",
            "returncode",
            "elapsed_sec",
            "timeout_sec",
            "score",
            "new_runs",
            "run_count",
            "start_ts",
            "end_ts",
            "stdout_tail",
            "env_json",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        total = len(tasks)
        done = 0
        for task in tasks:
            done += 1
            key = (task.model, task.combo_id, task.repeat)
            if key in completed_keys:
                print(f"[{done}/{total}] skip {task.model} {task.combo_id} rep={task.repeat} (resume)")
                continue

            spec = MODEL_SPECS[task.model]
            timeout_sec = int(max(1, math.ceil(spec.timeout_sec * args.timeout_scale)))
            start_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            print(f"[{done}/{total}] run {task.model} {task.combo_id} rep={task.repeat} timeout={timeout_sec}s")
            rc, elapsed, new_runs, score, tail = run_task(repo, runs_dir, task, timeout_sec)
            end_ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

            writer.writerow(
                {
                    "task_index": task.task_index,
                    "model": task.model,
                    "combo_id": task.combo_id,
                    "repeat": task.repeat,
                    "returncode": rc,
                    "elapsed_sec": f"{elapsed:.3f}",
                    "timeout_sec": timeout_sec,
                    "score": f"{score:.8g}" if math.isfinite(score) else "nan",
                    "new_runs": ";".join(new_runs),
                    "run_count": len(new_runs),
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "stdout_tail": tail.replace("\n", " | "),
                    "env_json": json.dumps(task.env, sort_keys=True),
                }
            )
            f.flush()

            score_msg = f"{score:.6f}" if math.isfinite(score) else "nan"
            print(
                f"  -> rc={rc} elapsed={elapsed:.1f}s runs={len(new_runs)} score={score_msg}"
            )

    print(f"Done. Summary: {summary_csv}")
    print(f"Plan: {plan_csv}")
    print(f"Meta: {meta_json}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise
