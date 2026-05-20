# SPDX-License-Identifier: Apache-2.0
"""Offline parity check: compare per-step loss trajectory between two runs.

Reads two JSONL files (one per run) and reports:
  - Pearson correlation of the chosen loss series
  - Per-bucket loss statistics
  - Buffer occupancy & injection rates (when present)

Intended use: capture an FV ``train_log.jsonl`` via JSONLLogCallback and an
upstream-equivalent JSONL with the same schema (``step``, ``total_loss``,
``svi_loss`` or ``train_loss``), then diff them here.

The verification gate for Stage 6 is loss-trajectory correlation >= 0.97 on
the first 200 steps under matched seeds/data/hyperparams. This script is
the comparator only; capturing the upstream trace is a separate manual step
(documented in ``docs/training/svi.md``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _series(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for r in rows:
        if key not in r:
            continue
        v = r[key]
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _pearson(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((ai - ma) * (bi - mb) for ai, bi in zip(a, b, strict=False))
    va = math.sqrt(sum((ai - ma)**2 for ai in a))
    vb = math.sqrt(sum((bi - mb)**2 for bi in b))
    if va == 0 or vb == 0:
        return float("nan")
    return cov / (va * vb)


def _stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean)**2 for x in xs) / max(1, n - 1)
    return {"n": float(n), "mean": mean, "std": math.sqrt(var), "min": min(xs), "max": max(xs)}


def main() -> None:
    parser = argparse.ArgumentParser(description="SVI training loss parity check")
    parser.add_argument("--a", required=True, help="JSONL from run A (e.g., FV)")
    parser.add_argument("--b", required=True, help="JSONL from run B (e.g., upstream)")
    parser.add_argument("--loss-key", default="total_loss",
                        help="Loss field name to compare (default: total_loss)")
    parser.add_argument("--min-correlation", type=float, default=0.97,
                        help="Pass threshold for Pearson correlation (default: 0.97)")
    parser.add_argument("--limit", type=int, default=0, help="Compare only the first N steps (0 = all)")
    args = parser.parse_args()

    a_rows = _load_jsonl(Path(args.a))
    b_rows = _load_jsonl(Path(args.b))
    if args.limit > 0:
        a_rows = a_rows[: args.limit]
        b_rows = b_rows[: args.limit]

    a_loss = _series(a_rows, args.loss_key)
    b_loss = _series(b_rows, args.loss_key)

    print(f"A: {Path(args.a).name}  rows={len(a_rows)}  loss-points={len(a_loss)}")
    print(f"B: {Path(args.b).name}  rows={len(b_rows)}  loss-points={len(b_loss)}")
    print()
    print(f"A {args.loss_key} stats: {_stats(a_loss)}")
    print(f"B {args.loss_key} stats: {_stats(b_loss)}")

    corr = _pearson(a_loss, b_loss)
    print()
    print(f"Pearson r({args.loss_key}) = {corr:.4f}  (threshold {args.min_correlation:.2f})")

    extra_keys = ("train/er_inject_noise", "train/er_inject_y", "train/er_inject_latent",
                  "train/er_noise_buffer_size", "train/er_y_buffer_size")
    for key in extra_keys:
        sa = _series(a_rows, key)
        sb = _series(b_rows, key)
        if not sa and not sb:
            continue
        print(f"  {key}:")
        print(f"    A {_stats(sa)}")
        print(f"    B {_stats(sb)}")

    if math.isnan(corr) or corr < args.min_correlation:
        print(f"\nFAIL: correlation {corr:.4f} below threshold {args.min_correlation:.2f}")
        sys.exit(1)
    print("\nPASS")


if __name__ == "__main__":
    main()
