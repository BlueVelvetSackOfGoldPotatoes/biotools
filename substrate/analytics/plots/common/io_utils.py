from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv_if_exists(path: Path | str, required_columns: Optional[Iterable[str]] = None) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p)
    if required_columns is not None:
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            return None
    return df


def latest_epoch(df: pd.DataFrame) -> int:
    if "epoch" not in df.columns or df.empty:
        return 0
    return int(df["epoch"].max())


def save_figure(fig, output_path: Path | str, fmt: str = "png") -> Path:
    out = Path(output_path)
    if out.suffix.lower() != f".{fmt.lower()}":
        out = out.with_suffix(f".{fmt.lower()}")
    ensure_dir(out.parent)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return out
