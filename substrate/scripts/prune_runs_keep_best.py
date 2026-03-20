#!/usr/bin/env python3
import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from random_search import score_run_dir


@dataclass
class RunInfo:
    run_id: str
    run_dir: Path
    family: str
    variant: str
    benchmark_id: str
    score: float
    train_end_utc: str
    mtime: float


def parse_iso_utc(ts: str) -> float:
    if not ts:
        return float("-inf")
    try:
        # Expected format: 2026-03-03T20:10:11Z
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")
        return dt.timestamp()
    except Exception:
        return float("-inf")


def finite(v: float) -> bool:
    return math.isfinite(v)


def run_family_from_manifest(manifest: Dict, run_id: str) -> str:
    fam = str(manifest.get("model_family") or "").strip().lower()
    if fam:
        return fam
    # Fallback for old/malformed manifests.
    return run_id.split("_", 1)[0].strip().lower() or "unknown"


def collect_runs(runs_dir: Path) -> List[RunInfo]:
    infos: List[RunInfo] = []
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        manifest_path = entry / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as exc:
            print(f"[warn] skipping {entry.name}: invalid manifest ({exc})", file=sys.stderr)
            continue

        score = score_run_dir(entry)
        family = run_family_from_manifest(manifest, entry.name)
        variant = str(manifest.get("model_variant") or "")
        benchmark_id = str(manifest.get("benchmark_id") or "")
        train_end_utc = str(manifest.get("train_end_utc") or "")
        mtime = entry.stat().st_mtime
        infos.append(
            RunInfo(
                run_id=entry.name,
                run_dir=entry,
                family=family,
                variant=variant,
                benchmark_id=benchmark_id,
                score=score,
                train_end_utc=train_end_utc,
                mtime=mtime,
            )
        )
    return infos


def pick_best_per_family(infos: List[RunInfo]) -> Dict[str, RunInfo]:
    by_family: Dict[str, List[RunInfo]] = defaultdict(list)
    for info in infos:
        by_family[info.family].append(info)

    winners: Dict[str, RunInfo] = {}
    for family, items in by_family.items():
        # Tie-break: higher score, then more recent train_end_utc, then mtime, then run_id.
        winner = max(
            items,
            key=lambda x: (
                x.score if finite(x.score) else float("-inf"),
                parse_iso_utc(x.train_end_utc),
                x.mtime,
                x.run_id,
            ),
        )
        winners[family] = winner
    return winners


def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete all run artifacts except the best run per architecture (model_family)."
    )
    ap.add_argument("--runs-dir", type=Path, default=Path("runs"))
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Perform deletion. Without this flag, only prints planned actions.",
    )
    ap.add_argument(
        "--report-csv",
        type=Path,
        default=Path("reports/prune_runs_keep_best_report.csv"),
        help="Write keep/delete decisions to CSV.",
    )
    args = ap.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.exists() or not runs_dir.is_dir():
        print(f"[error] runs dir not found: {runs_dir}", file=sys.stderr)
        return 2

    infos = collect_runs(runs_dir)
    if not infos:
        print("[info] no runs discovered")
        return 0

    winners = pick_best_per_family(infos)
    keep_ids = {w.run_id for w in winners.values()}
    to_delete = [info for info in infos if info.run_id not in keep_ids]
    to_keep = [info for info in infos if info.run_id in keep_ids]

    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.report_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "action",
                "run_id",
                "family",
                "variant",
                "benchmark_id",
                "score",
                "train_end_utc",
                "path",
            ]
        )
        for info in sorted(to_keep, key=lambda x: (x.family, x.run_id)):
            writer.writerow(
                [
                    "keep",
                    info.run_id,
                    info.family,
                    info.variant,
                    info.benchmark_id,
                    f"{info.score:.10g}" if finite(info.score) else "nan",
                    info.train_end_utc,
                    str(info.run_dir),
                ]
            )
        for info in sorted(to_delete, key=lambda x: (x.family, x.run_id)):
            writer.writerow(
                [
                    "delete",
                    info.run_id,
                    info.family,
                    info.variant,
                    info.benchmark_id,
                    f"{info.score:.10g}" if finite(info.score) else "nan",
                    info.train_end_utc,
                    str(info.run_dir),
                ]
            )

    print(f"[summary] total_runs={len(infos)} families={len(winners)} keep={len(to_keep)} delete={len(to_delete)}")
    print(f"[summary] report={args.report_csv.resolve()}")
    print("[kept]")
    for family in sorted(winners.keys()):
        w = winners[family]
        score_str = f"{w.score:.6f}" if finite(w.score) else "nan"
        print(f"  {family:18s} {w.run_id} score={score_str}")

    if not args.apply:
        print("[dry-run] no files deleted (use --apply to execute)")
        return 0

    reclaimed = 0
    for info in to_delete:
        if runs_dir not in info.run_dir.parents:
            print(f"[warn] refusing to delete outside runs dir: {info.run_dir}", file=sys.stderr)
            continue
        try:
            reclaimed += dir_size_bytes(info.run_dir)
            shutil.rmtree(info.run_dir)
        except Exception as exc:
            print(f"[warn] failed to delete {info.run_dir}: {exc}", file=sys.stderr)

    print(f"[done] deleted={len(to_delete)} kept={len(to_keep)} reclaimed_bytes={reclaimed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

