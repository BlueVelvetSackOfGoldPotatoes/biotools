from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.plots.common.common_plots import plot_error_gallery


def main() -> int:
    p = argparse.ArgumentParser(description="Plot error gallery summaries from run logs.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--max_pairs", type=int, default=12)
    args = p.parse_args()

    out = plot_error_gallery(args.run_dir, args.out_dir, args.format, args.max_pairs)
    print(f"generated {len(out)} error plot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
