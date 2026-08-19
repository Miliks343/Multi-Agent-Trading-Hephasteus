#!/usr/bin/env python3
"""Aggregate a finished suite into one CSV plus a printed table.

Reads each run's run_meta.json (args + provenance + wallclock) and, where eval
trajectories exist, computes the realised change in mark-to-market equity per
agent. Where fills were logged it also reports the spread-capture / adverse-
selection split - currently that is the F baseline path only, because
MarlChild does not log fills (see notes/design_space.md, E2).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def equity_change(npz):
    d = np.load(npz)
    eq = d["cash"].astype(np.int64) + d["inventory"].astype(np.int64) * d["mid_price"].astype(np.int64)
    return (eq[-1] - eq[0]) / 100.0


def markout(npz):
    """(capture, adverse_to_close) in dollars, or (None, None) without fills."""
    d = np.load(npz)
    if "fill_timestamps" not in d:
        return None, None
    ts, mid = d["timestamps"], d["mid_price"].astype(np.int64)
    ft, fs = d["fill_timestamps"], d["fill_side"].astype(np.int64)
    fp, fq = d["fill_price"].astype(np.int64), d["fill_quantity"].astype(np.int64)
    qi = np.clip(np.searchsorted(ts, ft, side="left") - 1, 0, len(mid) - 1)
    mid_q = mid[qi]
    capture = (fs * fq * (mid_q - fp)).sum() / 100.0
    adverse = (fs * fq * (mid[-1] - mid_q)).sum() / 100.0
    return capture, adverse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    root = repo / "runs" / args.suite
    if not root.exists():
        raise SystemExit(f"no runs found at {root.relative_to(repo)}")

    rows = []
    for meta_path in sorted(root.rglob("run_meta.json")):
        meta = json.loads(meta_path.read_text())
        a = meta["args"]
        run_dir = meta_path.parent
        row = {
            "run": str(run_dir.relative_to(root)),
            "n_agents": a.get("n_agents"),
            "value_agents": meta.get("config_kwargs", {}).get("num_value_agents", "default"),
            "seed": a.get("seed"),
            "min_size": a.get("min_size"),
            "max_offset": a.get("max_offset_cents"),
            "norm_obs": a.get("norm_obs"),
            "inv_penalty": a.get("inventory_penalty"),
            "timesteps": a.get("total_timesteps"),
            "train_min": round(meta.get("train_seconds", 0) / 60, 1),
            "host": meta.get("host"),
            "git_sha": (meta.get("git_sha") or "")[:8],
        }
        for npz in sorted((run_dir / "eval").glob("*trajectory*.npz")):
            tag = npz.stem.split("_")[-2]  # agent index
            row[f"d_equity_{tag}"] = round(equity_change(npz), 0)
            cap, adv = markout(npz)
            if cap is not None:
                row[f"capture_{tag}"] = round(cap, 0)
                row[f"adverse_{tag}"] = round(adv, 0)
        rows.append(row)

    if not rows:
        raise SystemExit("no run_meta.json found - did the suite finish?")

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "run", k))
    out = repo / "results" / f"{args.suite}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    show = [f for f in ("run", "n_agents", "value_agents", "seed", "norm_obs",
                        "train_min", "d_equity_0", "d_equity_1") if f in fields]
    widths = {f: max(len(f), max(len(str(r.get(f, ""))) for r in rows)) for f in show}
    print("  ".join(f.ljust(widths[f]) for f in show))
    print("-" * (sum(widths.values()) + 2 * (len(show) - 1)))
    for r in rows:
        print("  ".join(str(r.get(f, "")).ljust(widths[f]) for f in show))
    print(f"\nfull table ({len(fields)} columns) -> {out.relative_to(repo)}")


if __name__ == "__main__":
    main()
