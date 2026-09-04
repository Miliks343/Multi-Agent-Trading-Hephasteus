#!/usr/bin/env python3
"""Replay a suite's saved eval actions through the live quantisation.

Δequity says how much money a policy made; it does not say whether the policy
was making markets. This does. It takes the raw actions written to each eval
trajectory and pushes them through the same rounding that ``actions.py`` applies
before orders reach ABIDES, then reports how often a genuine two-sided quote
actually reached the book.

The distinction that matters: an action vector is not a quote. Offsets are
continuous cents rounded to whole ticks, sizes are rounded to integers, and
``translate_action`` voids *both* sides when the rounded bid would cross the
rounded ask. A policy emitting sub-tick offsets and sub-unit sizes therefore
posts nothing at all while still producing a full-looking action array.

    python scripts/quote_stats.py retrain
    python scripts/quote_stats.py grid --cells v10_n1,v400_n4
"""
import argparse
import glob
from collections import defaultdict
from pathlib import Path

import numpy as np

from marl_lob.actions import TICK_SIZE_CENTS


def _q_offset(raw: float) -> int:
    """Mirror actions._quantize_offset."""
    return int(round(max(float(raw), 0.0) / TICK_SIZE_CENTS)) * TICK_SIZE_CENTS


def _q_size(raw: float, max_size: int = 100) -> int:
    """Mirror actions._quantize_size."""
    return int(round(min(max(float(raw), 0.0), max_size)))


def cell_stats(npz_paths, max_size=100):
    agg = defaultdict(list)
    for f in npz_paths:
        d = np.load(f)
        if "actions" not in d:
            continue
        a = d["actions"]
        bid_off = np.array([_q_offset(x) for x in a[:, 0]])
        ask_off = np.array([_q_offset(x) for x in a[:, 1]])
        bid_qty = np.array([_q_size(x, max_size) for x in a[:, 2]])
        ask_qty = np.array([_q_size(x, max_size) for x in a[:, 3]])

        bid_ok, ask_ok = bid_qty > 0, ask_qty > 0
        # Prices are mid - bid_off and mid + ask_off, so the cross test
        # (bid_price >= ask_price) reduces to -bid_off >= ask_off.
        crossed = (bid_ok & ask_ok) & ((-bid_off) >= ask_off)
        two_sided = (bid_ok & ask_ok) & ~crossed
        any_quote = (bid_ok | ask_ok) & ~crossed

        agg["two_sided_pct"].append(two_sided.mean() * 100)
        agg["any_quote_pct"].append(any_quote.mean() * 100)
        agg["crossed_pct"].append(crossed.mean() * 100)
        agg["spread_cents"].append(
            float((bid_off + ask_off)[two_sided].mean()) if two_sided.any() else 0.0
        )
        agg["raw_offset"].append(float(a[:, :2].mean()))
        agg["raw_size"].append(float(a[:, 2:].mean()))
        agg["fills"].append(
            int(len(d["fill_timestamps"])) if "fill_timestamps" in d else 0
        )
    return {k: float(np.mean(v)) for k, v in agg.items()} if agg else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--cells", help="comma-separated subset; default all")
    ap.add_argument("--max-size", type=int, default=100)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / "runs" / args.suite
    if not root.is_dir():
        raise SystemExit(f"no runs at {root} - has the suite been run?")

    cells = (args.cells.split(",") if args.cells
             else sorted(p.name for p in root.iterdir() if p.is_dir()))

    hdr = ("cell", "two-sided%", "anyquote%", "crossed%", "spread(c)",
           "rawoff", "rawsize", "fills")
    print("%-16s%11s%11s%10s%11s%9s%9s%8s" % hdr)
    for cell in cells:
        paths = sorted(glob.glob(str(root / cell / "*" / "eval" / "*.npz")))
        if not paths:
            continue
        m = cell_stats(paths, args.max_size)
        if not m:
            continue
        print("%-16s%11.1f%11.1f%10.1f%11.2f%9.3f%9.2f%8.0f" % (
            cell, m["two_sided_pct"], m["any_quote_pct"], m["crossed_pct"],
            m["spread_cents"], m["raw_offset"], m["raw_size"], m["fills"]))


if __name__ == "__main__":
    main()
