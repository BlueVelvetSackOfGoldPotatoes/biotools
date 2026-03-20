from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.plots.common.common_plots import plot_confusion_matrix


def main() -> int:
    p = argparse.ArgumentParser(description="Plot confusion matrix from run logs.")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--format", default="png")
    p.add_argument("--split", default="test")
    p.add_argument("--epoch", type=int, default=None)
    args = p.parse_args()

    out = plot_confusion_matrix(args.run_dir, args.out_dir, args.format, args.split, args.epoch)
    print(f"generated {len(out)} confusion plot(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
