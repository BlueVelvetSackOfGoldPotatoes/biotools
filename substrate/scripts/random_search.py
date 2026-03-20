#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple


def log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def finite_or_none(v: str):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def score_run_dir(run_dir: Path) -> float:
    epoch_csv = run_dir / "learning" / "epoch_metrics.csv"
    if not epoch_csv.exists():
        return float("-inf")

    best_test_acc = float("-inf")
    best_train_acc = float("-inf")
    best_neg_test_loss = float("-inf")

    with epoch_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = (row.get("split") or "").strip().lower()
            acc = finite_or_none(row.get("accuracy", ""))
            loss = finite_or_none(row.get("loss", ""))

            if split == "test":
                if acc is not None:
                    best_test_acc = max(best_test_acc, acc)
                if loss is not None:
                    best_neg_test_loss = max(best_neg_test_loss, -loss)
            if split == "train" and acc is not None:
                best_train_acc = max(best_train_acc, acc)

    if best_test_acc > float("-inf"):
        return best_test_acc
    if best_train_acc > float("-inf"):
        return best_train_acc
    if best_neg_test_loss > float("-inf"):
        return best_neg_test_loss
    return float("-inf")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


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


def sample_mlp(rng: random.Random) -> Dict[str, str]:
    return {
        "MLP_EPOCHS": str(rng.choice([6, 10, 14])),
        "MLP_BATCH_SIZE": str(rng.choice([32, 64, 128])),
        "MLP_LR": f"{log_uniform(rng, 1e-4, 3e-2):.8g}",
        "MLP_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "MLP_LOSS": rng.choice(["cross_entropy", "mse"]),
        "MLP_H1": str(rng.choice([128, 256, 384])),
        "MLP_H2": str(rng.choice([64, 128, 192])),
    }


def sample_cnn(rng: random.Random) -> Dict[str, str]:
    return {
        "CNN_VARIANT": rng.choice(["simple_conv", "lenet5", "resnet_simple"]),
        "CNN_EPOCHS": str(rng.choice([4, 6, 8])),
        "CNN_BATCH_SIZE": str(rng.choice([32, 64])),
        "CNN_LR": f"{log_uniform(rng, 1e-4, 8e-3):.8g}",
        "CNN_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "CNN_LOSS": rng.choice(["cross_entropy", "mse"]),
    }


def sample_rnn(rng: random.Random) -> Dict[str, str]:
    return {
        "RNN_VARIANT": rng.choice(["vanilla_rnn", "gru", "birnn"]),
        "RNN_EPOCHS": str(rng.choice([6, 10, 14])),
        "RNN_BATCH_SIZE": str(rng.choice([32, 64, 128])),
        "RNN_LR": f"{log_uniform(rng, 3e-4, 2e-2):.8g}",
        "RNN_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "RNN_LOSS": rng.choice(["cross_entropy", "mse"]),
        "RNN_HIDDEN": str(rng.choice([64, 128, 192])),
        "RNN_GRU_HIDDEN": str(rng.choice([64, 128, 192])),
        "RNN_BI_HIDDEN": str(rng.choice([48, 64, 96])),
    }


def sample_lstm(rng: random.Random) -> Dict[str, str]:
    return {
        "LSTM_VARIANT": rng.choice(["lstm", "bilstm"]),
        "LSTM_EPOCHS": str(rng.choice([6, 10, 14])),
        "LSTM_BATCH_SIZE": str(rng.choice([32, 64, 128])),
        "LSTM_LR": f"{log_uniform(rng, 3e-4, 2e-2):.8g}",
        "LSTM_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "LSTM_LOSS": rng.choice(["cross_entropy", "mse"]),
        "LSTM_HIDDEN": str(rng.choice([64, 128, 192])),
        "LSTM_BI_HIDDEN": str(rng.choice([48, 64, 96])),
    }


def sample_transformer(rng: random.Random) -> Dict[str, str]:
    heads = rng.choice([2, 4, 8])
    d_model = rng.choice([64, 96, 128, 160, 192])
    d_model = ((d_model + heads - 1) // heads) * heads
    return {
        "TRANSFORMER_EPOCHS": str(rng.choice([6, 10, 14])),
        "TRANSFORMER_BATCH_SIZE": str(rng.choice([32, 64])),
        "TRANSFORMER_LR": f"{log_uniform(rng, 1e-4, 8e-3):.8g}",
        "TRANSFORMER_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "TRANSFORMER_LOSS": rng.choice(["cross_entropy", "mse"]),
        "TRANSFORMER_D_MODEL": str(d_model),
        "TRANSFORMER_NUM_HEADS": str(heads),
        "TRANSFORMER_D_FF": str(rng.choice([128, 192, 256, 384])),
        "TRANSFORMER_NUM_BLOCKS": str(rng.choice([1, 2, 3])),
        "TRANSFORMER_DROPOUT": f"{rng.choice([0.0, 0.05, 0.1, 0.15]):.3g}",
    }


def sample_vit(rng: random.Random) -> Dict[str, str]:
    heads = rng.choice([2, 4, 8])
    embed_dim = rng.choice([64, 96, 128, 160, 192])
    embed_dim = ((embed_dim + heads - 1) // heads) * heads
    return {
        "VIT_EPOCHS": str(rng.choice([6, 10, 14])),
        "VIT_BATCH_SIZE": str(rng.choice([32, 64])),
        "VIT_LR": f"{log_uniform(rng, 1e-4, 5e-3):.8g}",
        "VIT_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "VIT_LOSS": rng.choice(["cross_entropy", "mse"]),
        "VIT_PATCH_SIZE": str(rng.choice([4, 7, 14])),
        "VIT_EMBED_DIM": str(embed_dim),
        "VIT_NUM_HEADS": str(heads),
        "VIT_D_FF": str(rng.choice([128, 192, 256, 384])),
        "VIT_NUM_BLOCKS": str(rng.choice([1, 2, 3])),
        "VIT_DROPOUT": f"{rng.choice([0.0, 0.05, 0.1, 0.15]):.3g}",
    }


def sample_hebbian(rng: random.Random) -> Dict[str, str]:
    return {
        "HEBBIAN_RULE": rng.choice(["hebb", "sanger", "bcm"]),
        "HEBBIAN_LR": f"{log_uniform(rng, 1e-3, 3e-2):.8g}",
        "HEBBIAN_PASSES": str(rng.choice([4, 8, 12])),
        "HEBBIAN_FEATURE_WIDTH": str(rng.choice([256, 384, 512])),
        "HEBBIAN_PROBE_EPOCHS": str(rng.choice([30, 60, 100])),
        "HEBBIAN_PROBE_LR": f"{log_uniform(rng, 5e-5, 5e-3):.8g}",
        "HEBBIAN_PROBE_OPTIMIZER": rng.choice(["adam", "sgd", "rmsprop"]),
        "HEBBIAN_PROBE_LOSS": rng.choice(["cross_entropy", "mse"]),
    }


def sample_ff(rng: random.Random) -> Dict[str, str]:
    return {
        "FF_EPOCHS": str(rng.choice([20, 40, 60])),
        "FF_BATCH_SIZE": str(rng.choice([32, 64, 128])),
        "FF_LR": f"{log_uniform(rng, 5e-3, 8e-2):.8g}",
        "FF_THRESHOLD": f"{rng.choice([1.0, 1.5, 2.0, 2.5, 3.0]):.3g}",
        "FF_HIDDEN1": str(rng.choice([256, 384, 500, 640])),
        "FF_HIDDEN2": str(rng.choice([256, 384, 500, 640])),
        "FF_TRAIN_N": str(rng.choice([15000, 30000, 60000])),
    }


def sample_gnn(rng: random.Random) -> Dict[str, str]:
    return {
        "GNN_EPOCHS": str(rng.choice([10, 20, 30])),
        "GNN_LR": f"{log_uniform(rng, 5e-4, 8e-3):.8g}",
        "GNN_HIDDEN": str(rng.choice([64, 96, 128, 192])),
        "GNN_LAYERS": str(rng.choice([2, 3, 4, 5])),
        "GNN_TRAIN_N": str(rng.choice([5000, 10000, 20000])),
        "GNN_MID_TEST_N": str(rng.choice([1000, 2000, 4000])),
    }


def sample_diffusion(rng: random.Random) -> Dict[str, str]:
    return {
        "DIFFUSION_STEPS": str(rng.choice([1000, 3000, 6000])),
        "DIFFUSION_BATCH_SIZE": str(rng.choice([64, 96, 128])),
        "DIFFUSION_LR": f"{log_uniform(rng, 1e-4, 3e-3):.8g}",
        "DIFFUSION_TIMESTEPS": str(rng.choice([10, 20, 30])),
        "DIFFUSION_HIDDEN": str(rng.choice([256, 384, 512])),
    }


def sample_actor_critic(rng: random.Random) -> Dict[str, str]:
    return {
        "ACTOR_CRITIC_STEPS": str(rng.choice([4000, 8000, 16000])),
        "ACTOR_CRITIC_BATCH": str(rng.choice([64, 128, 256])),
        "ACTOR_CRITIC_HIDDEN": str(rng.choice([128, 192, 256])),
        "ACTOR_CRITIC_GAMMA": f"{rng.choice([0.0, 0.9, 0.99]):.3g}",
        "A2C_LR_POLICY": f"{log_uniform(rng, 1e-4, 3e-3):.8g}",
        "A2C_LR_VALUE": f"{log_uniform(rng, 1e-4, 3e-3):.8g}",
        "PPO_LR": f"{log_uniform(rng, 1e-4, 1e-3):.8g}",
        "PPO_CLIP_EPS": f"{rng.choice([0.1, 0.2, 0.3]):.3g}",
        "ACTOR_CRITIC_EVAL_EVERY": str(rng.choice([1000, 2000, 4000])),
    }


def sample_reinforcement(rng: random.Random) -> Dict[str, str]:
    algo = rng.choice(["q_learning", "sarsa", "dqn", "ddqn", "muzero_lite"])
    env = {
        "RL_ALGORITHMS": algo,
        "RL_TRAIN_N": str(rng.choice([8000, 12000, 20000])),
        "RL_TEST_N": str(rng.choice([1000, 2000])),
    }
    if algo in ("q_learning", "sarsa"):
        prefix = "RL_Q" if algo == "q_learning" else "RL_SARSA"
        env[f"{prefix}_STEPS"] = str(rng.choice([3000, 6000, 12000]))
        env[f"{prefix}_ALPHA"] = f"{log_uniform(rng, 0.03, 0.4):.8g}"
        env[f"{prefix}_GAMMA"] = f"{rng.choice([0.0, 0.5, 0.9]):.3g}"
    elif algo in ("dqn", "ddqn"):
        env["RL_DQN_STEPS"] = str(rng.choice([2000, 4000, 8000]))
        env["RL_DQN_HIDDEN"] = str(rng.choice([128, 192, 256]))
        env["RL_DQN_LR"] = f"{log_uniform(rng, 5e-5, 2e-3):.8g}"
        env["RL_DQN_BATCH"] = str(rng.choice([32, 64, 128]))
        env["RL_DQN_REPLAY"] = str(rng.choice([20000, 50000, 100000]))
    else:
        env["RL_MUZERO_STEPS"] = str(rng.choice([1500, 3000, 5000]))
        env["RL_MUZERO_LATENT"] = str(rng.choice([32, 64, 96]))
        env["RL_MUZERO_LR"] = f"{log_uniform(rng, 1e-4, 2e-3):.8g}"
        env["RL_MUZERO_SIMULATIONS"] = str(rng.choice([8, 15, 25]))
        env["RL_MUZERO_BATCH"] = str(rng.choice([32, 64, 128]))
    return env


def sample_markov(rng: random.Random) -> Dict[str, str]:
    return {
        "MARKOV_ORDER": str(rng.choice([1, 2, 3])),
        "MARKOV_BINS": str(rng.choice([4, 6, 8, 12])),
        "MARKOV_ALPHA": f"{log_uniform(rng, 1e-3, 1.0):.8g}",
        "MARKOV_EPOCHS": str(rng.choice([4, 6, 8, 12])),
        "MARKOV_TRAIN_N": str(rng.choice([10000, 20000, 30000, 60000])),
        "MARKOV_TEST_N": str(rng.choice([2000, 5000, 10000])),
        "MARKOV_NLL_EVAL_N": str(rng.choice([256, 512, 1024])),
        "MARKOV_SHUFFLE": rng.choice(["1", "0"]),
    }


def sample_tictactoe(rng: random.Random) -> Dict[str, str]:
    model = rng.choice(
        [
            "mlp",
            "cnn",
            "rnn",
            "gru",
            "lstm",
            "transformer",
            "vit",
            "hebbian",
            "actor_critic",
            "hybrid:cnn,transformer,mlp",
            "hybrid:actor_critic,hebbian,mlp",
            "hybrid:gru,lstm,mlp",
        ]
    )
    return {
        "TICTACTOE_ALGORITHM": "deep_q",
        "TICTACTOE_MODEL": model,
        "TICTACTOE_EPISODES": str(rng.choice([6000, 12000, 20000])),
        "TICTACTOE_EVAL_EVERY": str(rng.choice([600, 1000, 1500])),
        "TICTACTOE_EVAL_EPISODES": str(rng.choice([200, 300, 500])),
        "TICTACTOE_LR": f"{log_uniform(rng, 5e-4, 5e-3):.8g}",
        "TICTACTOE_GAMMA": f"{rng.choice([0.90, 0.94, 0.97, 0.99]):.3g}",
        "TICTACTOE_EPS_START": f"{rng.choice([0.9, 1.0]):.3g}",
        "TICTACTOE_EPS_MIN": f"{rng.choice([0.01, 0.03, 0.05]):.3g}",
        "TICTACTOE_EPS_DECAY": f"{rng.choice([0.9993, 0.9995, 0.9997, 0.99985]):.6g}",
        "TICTACTOE_OPP_NOISE_TRAIN": f"{rng.choice([0.05, 0.10, 0.15, 0.20]):.3g}",
        "TICTACTOE_OPP_NOISE_EVAL": f"{rng.choice([0.0, 0.02, 0.05]):.3g}",
        "TICTACTOE_BRIDGE_WIDTH": str(rng.choice([384, 512, 784, 1024])),
        "TICTACTOE_GRAD_CLIP": f"{rng.choice([1.0, 3.0, 5.0, 8.0]):.3g}",
    }


def sample_trees(rng: random.Random) -> Dict[str, str]:
    return {
        "TREE_DEPTHS": str(rng.choice([8, 12, 20, 30, 50])),
        "TREE_MIN_SAMPLES_SPLIT": str(rng.choice([2, 5, 10, 20])),
    }


def sample_forests(rng: random.Random) -> Dict[str, str]:
    return {
        "FORESTS_VARIANT": rng.choice(["random_forest", "extra_trees"]),
        "FORESTS_N_TREES": str(rng.choice([20, 50, 100])),
        "FORESTS_MAX_DEPTH": str(rng.choice([8, 12, 15, 20, 30])),
        "FORESTS_MIN_SAMPLES_SPLIT": str(rng.choice([2, 5, 10])),
        "FORESTS_RF_SAMPLE_RATIO": f"{rng.choice([0.6, 0.8, 1.0]):.3g}",
    }


def sample_clustering(rng: random.Random) -> Dict[str, str]:
    return {
        "CLUSTERING_ALGORITHM": rng.choice(["kmeans", "minibatch_kmeans", "gmm", "agglomerative"]),
        "CLUSTERING_K": str(rng.choice([8, 10, 12])),
        "CLUSTERING_MAX_ITER": str(rng.choice([50, 100, 150])),
        "CLUSTERING_MINIBATCH_SIZE": str(rng.choice([64, 128, 256])),
        "CLUSTERING_N": str(rng.choice([2000, 3000, 5000])),
        "CLUSTERING_AGG_N": str(rng.choice([300, 500, 700])),
    }


def sample_continuous(rng: random.Random) -> Dict[str, str]:
    mode = rng.choice(["single", "hybrid"])
    env = {
        "ONLINE_MODE": mode,
        "ONLINE_CYCLES": str(rng.choice([6, 10, 14])),
        "ONLINE_ACTIVE_STEPS": str(rng.choice([8, 16, 24])),
        "ONLINE_SLEEP_STEPS": str(rng.choice([4, 8, 12])),
        "ONLINE_ACTIVE_BATCH": str(rng.choice([32, 64, 96])),
        "ONLINE_SLEEP_BATCH": str(rng.choice([24, 48, 64])),
        "ONLINE_LR": f"{log_uniform(rng, 5e-4, 5e-3):.8g}",
        "ONLINE_EVAL_SUBSET": str(rng.choice([1000, 2000, 4000])),
        "ONLINE_PASS_ACC": f"{rng.choice([0.65, 0.7, 0.75]):.3g}",
    }
    if mode == "single":
        env["ONLINE_MODEL"] = rng.choice(["mlp", "cnn", "gru", "lstm", "transformer", "vit"])
    else:
        env["ONLINE_HYBRID_SPEC"] = rng.choice(
            [
                "cnn,mlp",
                "cnn,transformer,mlp",
                "gru,lstm,mlp",
                "vit,transformer,cnn",
            ]
        )
    return env


@dataclass
class ModelSpec:
    name: str
    binary: str
    sampler: Callable[[random.Random], Dict[str, str]]
    timeout_sec: int


MODEL_SPECS: Dict[str, ModelSpec] = {
    "mlp": ModelSpec("mlp", "bin/benchmark_mlp", sample_mlp, 600),
    "cnn": ModelSpec("cnn", "bin/benchmark_cnn", sample_cnn, 900),
    "rnn": ModelSpec("rnn", "bin/benchmark_rnn", sample_rnn, 900),
    "lstm": ModelSpec("lstm", "bin/benchmark_lstm", sample_lstm, 900),
    "transformer": ModelSpec("transformer", "bin/benchmark_transformer", sample_transformer, 1200),
    "vit": ModelSpec("vit", "bin/benchmark_vit", sample_vit, 1200),
    "hebbian": ModelSpec("hebbian", "bin/benchmark_hebbian", sample_hebbian, 1200),
    "forward_forward": ModelSpec("forward_forward", "bin/benchmark_forward_forward", sample_ff, 900),
    "gnn": ModelSpec("gnn", "bin/benchmark_gnn", sample_gnn, 900),
    "diffusion": ModelSpec("diffusion", "bin/benchmark_diffusion", sample_diffusion, 1800),
    "actor_critic": ModelSpec("actor_critic", "bin/benchmark_actor_critic", sample_actor_critic, 1800),
    "reinforcement": ModelSpec("reinforcement", "bin/benchmark_reinforcement", sample_reinforcement, 1800),
    "markov": ModelSpec("markov", "bin/benchmark_markov", sample_markov, 1200),
    "tictactoe": ModelSpec("tictactoe", "bin/benchmark_tictactoe", sample_tictactoe, 900),
    "trees": ModelSpec("trees", "bin/benchmark_trees", sample_trees, 600),
    "forests": ModelSpec("forests", "bin/benchmark_forests", sample_forests, 900),
    "clustering": ModelSpec("clustering", "bin/benchmark_clustering", sample_clustering, 600),
    "continuous": ModelSpec("continuous", "bin/benchmark_continuous", sample_continuous, 1800),
}


def run_trial(
    repo: Path,
    runs_dir: Path,
    spec: ModelSpec,
    trial_id: int,
    seed: int,
    extra_env: Dict[str, str],
) -> Tuple[float, Dict[str, str], int, float, List[str], str]:
    rng = random.Random(seed + trial_id * 10007)
    sampled_env = spec.sampler(rng)
    sampled_env.update(extra_env)
    trial_uuid = f"rs_{spec.name}_{trial_id}_{uuid.uuid4().hex[:10]}"
    sampled_env["TRIAL_UUID"] = trial_uuid
    sampled_env.setdefault("JOB_ORIGIN", "random_search")

    before = set(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else set()
    cmd = [str(repo / spec.binary)]
    env = os.environ.copy()
    env.update(sampled_env)

    t0 = time.time()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as out:
        proc = subprocess.run(
            cmd,
            cwd=str(repo),
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=spec.timeout_sec,
        )
        elapsed = time.time() - t0

        out.seek(0)
        tail_buf = deque(maxlen=8)
        for line in out:
            tail_buf.append(line.rstrip("\n"))
        tail = "\n".join(tail_buf)

    after = set(p.name for p in runs_dir.iterdir() if p.is_dir()) if runs_dir.exists() else set()
    new_runs = sorted(after - before)
    owned_runs = runs_for_trial_uuid(runs_dir, new_runs, trial_uuid)
    if not owned_runs and new_runs:
        # Backward-compatible fallback if a benchmark does not yet emit trial_uuid.
        owned_runs = new_runs

    score = float("-inf")
    for run_id in owned_runs:
        run_score = score_run_dir(runs_dir / run_id)
        if run_score > score:
            score = run_score

    return score, sampled_env, proc.returncode, elapsed, owned_runs, tail


def main():
    ap = argparse.ArgumentParser(description="Random hyperparameter search over benchmark binaries.")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--models", type=str, default="all", help="comma-separated model keys or 'all'")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--extra-env",
        type=str,
        default="",
        help="comma-separated KEY=VALUE env pairs applied to all trials",
    )
    args = ap.parse_args()

    repo = args.repo.resolve()
    runs_dir = repo / "runs"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (repo / "reports" / f"hpo_{timestamp}")
    ensure_dir(out_dir)

    extra_env: Dict[str, str] = {}
    if args.extra_env.strip():
        for item in args.extra_env.split(","):
            if not item.strip():
                continue
            if "=" not in item:
                raise ValueError(f"Invalid --extra-env entry '{item}', expected KEY=VALUE")
            k, v = item.split("=", 1)
            extra_env[k.strip()] = v.strip()

    if args.models.strip().lower() == "all":
        model_keys = list(MODEL_SPECS.keys())
    else:
        model_keys = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Available: {sorted(MODEL_SPECS.keys())}")

    summary_rows = []
    best_by_model = {}

    for model_idx, key in enumerate(model_keys):
        spec = MODEL_SPECS[key]
        print(f"\n=== Random Search: {key} ({args.trials} trials) ===")
        best_score = float("-inf")
        best_row = None

        for trial in range(1, args.trials + 1):
            trial_seed = args.seed + model_idx * 100000 + trial
            try:
                score, env_cfg, returncode, elapsed, new_runs, stdout_tail = run_trial(
                    repo, runs_dir, spec, trial, trial_seed, extra_env
                )
            except subprocess.TimeoutExpired:
                score = float("-inf")
                env_cfg = spec.sampler(random.Random(trial_seed + trial * 10007))
                env_cfg.update(extra_env)
                returncode = 124
                elapsed = float(spec.timeout_sec)
                new_runs = []
                stdout_tail = "timeout"

            row = {
                "model": key,
                "trial": trial,
                "score": score,
                "returncode": returncode,
                "elapsed_sec": elapsed,
                "new_runs": ";".join(new_runs),
                "env_json": json.dumps(env_cfg, sort_keys=True),
                "stdout_tail": stdout_tail.replace("\n", " | "),
            }
            summary_rows.append(row)

            score_str = f"{score:.6f}" if math.isfinite(score) else "nan"
            print(
                f"trial {trial:02d}/{args.trials} | score={score_str} | rc={returncode} | "
                f"elapsed={elapsed:.1f}s | runs={len(new_runs)}"
            )

            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None:
            best_by_model[key] = best_row

    summary_csv = out_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "trial",
                "score",
                "returncode",
                "elapsed_sec",
                "new_runs",
                "env_json",
                "stdout_tail",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    best_json = out_dir / "best_configs.json"
    with best_json.open("w") as f:
        json.dump(best_by_model, f, indent=2, sort_keys=True)

    print("\n=== Best Configs ===")
    for key in model_keys:
        row = best_by_model.get(key)
        if not row:
            print(f"{key}: no valid trials")
            continue
        score = row["score"]
        score_str = f"{score:.6f}" if isinstance(score, (int, float)) and math.isfinite(score) else "nan"
        print(f"{key}: score={score_str} trial={row['trial']} rc={row['returncode']}")

    print(f"\nWrote summary: {summary_csv}")
    print(f"Wrote best configs: {best_json}")


if __name__ == "__main__":
    main()
