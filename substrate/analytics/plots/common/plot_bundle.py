from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.plots.common.bundle import generate_common_bundle, generate_model_bundle, detect_model_family


def main() -> int:
    p = argparse.ArgumentParser(description="Generate all common + model-specific plots for a run.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--model_family", default=None)
    args = p.parse_args()

    common = generate_common_bundle(args.run_dir, args.out_dir, args.format)
    family = args.model_family or detect_model_family(args.run_dir)
    model = generate_model_bundle(args.run_dir, args.out_dir, args.format, family)
    print(f"model_family={family} generated={len(common) + len(model)} plot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
