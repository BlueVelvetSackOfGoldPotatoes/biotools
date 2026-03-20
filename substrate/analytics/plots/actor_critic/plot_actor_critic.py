from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.plots.common.bundle import generate_common_bundle, generate_model_bundle


MODEL_FAMILY = "actor_critic"


def main() -> int:
    p = argparse.ArgumentParser(description=f"Generate plots for {MODEL_FAMILY} runs.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", default="png")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    common = generate_common_bundle(args.run_dir, out_dir / "common", args.format)
    model = generate_model_bundle(args.run_dir, out_dir / MODEL_FAMILY, args.format, MODEL_FAMILY)
    print(f"{MODEL_FAMILY}: generated {len(common)} common + {len(model)} model-specific plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
