from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

MODELS = [
    "mlp", "cnn", "rnn", "lstm", "transformer", "vit", "trees", "forests",
    "clustering", "hebbian", "actor_critic", "reinforcement", "diffusion", "gnn", "forward_forward",
]


def main() -> int:
    p = argparse.ArgumentParser(description="Generate plot bundles for all model scripts.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", default="png")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_dir)

    for m in MODELS:
        script = Path(__file__).resolve().parent / m / f"plot_{m}.py"
        if not script.exists():
            continue
        out_dir = out_root / m
        cmd = [sys.executable, str(script), "--run_dir", str(run_dir), "--out_dir", str(out_dir), "--format", args.format]
        subprocess.run(cmd, check=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
