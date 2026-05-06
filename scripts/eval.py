"""Roll a trained PPO checkpoint through MarlLobEnv and compare against F.

Side-by-side metric printout per (seed, agent), reading F's saved
trajectories from `runs/baseline/`. The "one command, two Sharpe numbers"
deliverable for chunk 9.

Assumes scripts/run_baseline.py has been run on the same seeds first
(otherwise: pass --no-baseline to just print PPO numbers).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from marl_lob.env import MarlLobEnv
from marl_lob.metrics import compute_all
from marl_lob.trajectory import Fill, Trajectory

warnings.filterwarnings("ignore")


def rollout_ppo(env: MarlLobEnv, model, seed: int, max_steps: int) -> dict[str, Trajectory]:
    """Run a deterministic-policy rollout, return per-agent Trajectory."""
    obs, info = env.reset(seed=seed)
    rows: dict[str, list[tuple]] = {
        a: [info[a]["traj_row"]] for a in env.possible_agents
    }
    for _ in range(max_steps):
        if not env.agents:
            break
        actions = {
            a: model.predict(obs[a], deterministic=True)[0].astype(np.float32)
            for a in env.agents
        }
        obs, _r, _t, trunc, info = env.step(actions)
        for a in env.possible_agents:
            if a in info:
                rows[a].append(info[a]["traj_row"])
        if any(trunc.values()):
            break
    return {a: Trajectory.from_tuples(r) for a, r in rows.items()}


def load_baseline_trajectory(path: Path) -> Trajectory:
    """Reconstruct a Trajectory (incl. fills) from the .npz saved by run_baseline."""
    d = np.load(path)
    fills = [
        Fill(
            timestamp=float(d["fill_timestamps"][i]),
            side=int(d["fill_side"][i]),
            price=int(d["fill_price"][i]),
            quantity=int(d["fill_quantity"][i]),
        )
        for i in range(len(d["fill_timestamps"]))
    ]
    return Trajectory(
        timestamps=d["timestamps"],
        inventory=d["inventory"],
        cash=d["cash"],
        mid_price=d["mid_price"],
        fills=fills,
    )


def infer_dt_seconds(traj: Trajectory) -> float:
    """Median Δt between snapshots — used to annualize Sharpe."""
    if len(traj) < 2:
        return 1.0
    return float(np.median(np.diff(traj.timestamps)))


def metric_block(label: str, traj: Trajectory) -> None:
    if len(traj) == 0:
        print(f"    {label}: no snapshots")
        return
    dt = infer_dt_seconds(traj)
    m = compute_all(traj, dt_seconds=dt)
    eq = traj.equity()
    print(
        f"    {label:>5}: snaps={len(traj):>4}  fills={len(traj.fills):>4}  "
        f"Sharpe={m['sharpe']:>+8.2f}  "
        f"MaxDD={m['max_drawdown'] * 100:>6.2f}%  "
        f"Δeq={int(eq[-1] - eq[0]):>+10d}c  "
        f"finalInv={int(traj.inventory[-1]):>+5d}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path,
                        help="path to a .zip checkpoint from scripts/train.py")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--n-agents", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--baseline-dir", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/eval"))
    parser.add_argument("--no-baseline", action="store_true",
                        help="skip F comparison (just print PPO numbers)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = PPO.load(args.checkpoint)
    print(f"loaded checkpoint: {args.checkpoint}")

    for seed in args.seeds:
        env = MarlLobEnv(n_agents=args.n_agents, max_inventory=10_000)
        ppo_trajs = rollout_ppo(env, model, seed, args.max_steps)
        env.close()

        print(f"\n== seed {seed} ==")
        for i in range(args.n_agents):
            agent = f"mm_{i}"
            ppo = ppo_trajs[agent]
            print(f"  agent {agent}")
            metric_block("PPO", ppo)

            if not args.no_baseline:
                f_path = args.baseline_dir / f"trajectory_{i}_seed{seed}.npz"
                if f_path.exists():
                    f_traj = load_baseline_trajectory(f_path)
                    metric_block("F", f_traj)
                else:
                    print(f"    F   : no baseline at {f_path}; "
                          f"run scripts/run_baseline.py --seed {seed}")

            # Save PPO trajectory for later analysis
            np.savez(
                args.out_dir / f"ppo_trajectory_{i}_seed{seed}.npz",
                timestamps=ppo.timestamps,
                inventory=ppo.inventory,
                cash=ppo.cash,
                mid_price=ppo.mid_price,
            )

    print(f"\nPPO trajectories saved → {args.out_dir}/")


if __name__ == "__main__":
    main()
