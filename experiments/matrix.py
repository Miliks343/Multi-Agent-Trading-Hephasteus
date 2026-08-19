#!/usr/bin/env python3
"""Single source of truth for the experiment suites.

Emits one executable shell script per job into experiments/jobs/<suite>/ so the
runner can simply parallelise over files, each job gets its own log, and any job
can be re-run by hand. Nothing here runs anything.

Suites:
  smoke  1 short run - proves the pipeline works on a new machine
  cheap  the four high-value cheap experiments (see notes/design_space.md)
  grid   the 2D spine: informed fraction x number of market makers
"""
import argparse
import os
import shlex
import stat
from pathlib import Path

SEEDS = (0, 1, 7)


def job(name, train_args, *, n_agents=2, eval_seeds=(42,), timesteps=50_000):
    return {"name": name, "n_agents": n_agents, "timesteps": timesteps,
            "train_args": list(train_args), "eval_seeds": list(eval_seeds)}


def suite_smoke():
    return [job("smoke", [], timesteps=2_000)]


def suite_cheap():
    jobs = []
    # E1: observations are raw cents (~1e5). Same magnitude problem as the
    # reward, never tested on the input side.
    for s in SEEDS:
        jobs.append(job(f"norm_obs/seed{s}", ["--norm-obs", "--seed", s]))
        jobs.append(job(f"norm_obs_off/seed{s}", ["--seed", s]))
    # Close BOTH escape hatches at once. min_size alone was routed around by
    # quoting ~50c off mid; clamp the offset at the same time.
    for s in SEEDS:
        jobs.append(job(f"both_hatches/seed{s}",
                        ["--min-size", 10, "--max-offset-cents", 5, "--seed", s]))
    # Restore the env's own inventory penalty, which the ablation showed is
    # strictly better than removing it when VecNormalize is on.
    for s in SEEDS:
        jobs.append(job(f"invpen_default/seed{s}",
                        ["--inventory-penalty", 1e-4, "--seed", s]))
    return jobs


def suite_grid():
    """informed fraction x competition.

    Predictions, committed before running (notes/design_space.md):
      - more informed flow  -> wider quotes, fewer fills, eventual withdrawal
      - more competitors    -> tighter quotes, less profit each
      - interaction: competition removes the cushion that lets a maker survive
        adverse selection, so withdrawal arrives at a LOWER informed fraction
        as n_agents rises
    """
    jobs = []
    for value_agents in (10, 50, 150, 400):
        for n_agents in (1, 2, 4):
            for s in SEEDS:
                jobs.append(job(
                    f"v{value_agents}_n{n_agents}/seed{s}",
                    ["--num-value-agents", value_agents, "--seed", s],
                    n_agents=n_agents,
                ))
    return jobs


SUITES = {"smoke": suite_smoke, "cheap": suite_cheap, "grid": suite_grid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=sorted(SUITES))
    ap.add_argument("--jobs-dir", type=Path, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    jobs_dir = args.jobs_dir or repo / "experiments" / "jobs" / args.suite
    if jobs_dir.exists():
        for f in jobs_dir.glob("*.sh"):
            f.unlink()
    jobs_dir.mkdir(parents=True, exist_ok=True)

    jobs = SUITES[args.suite]()
    for i, j in enumerate(jobs):
        out = f"runs/{args.suite}/{j['name']}"
        train = ["python", "scripts/train.py",
                 "--out-dir", out,
                 "--n-agents", str(j["n_agents"]),
                 "--total-timesteps", str(j["timesteps"])]
        train += [str(a) for a in j["train_args"]]
        ev = ["python", "scripts/eval.py", f"{out}/ppo_marl_lob.zip",
              "--n-agents", str(j["n_agents"]),
              "--seeds", *[str(s) for s in j["eval_seeds"]],
              "--out-dir", f"{out}/eval", "--no-baseline"]

        flat = j["name"].replace("/", "__")
        script = jobs_dir / f"{i:03d}_{flat}.sh"
        script.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f'cd "$(dirname "${{BASH_SOURCE[0]}}")/../../.."\n'
            f'echo "=== {j["name"]} ==="\n'
            f"{shlex.join(train)}\n"
            f"{shlex.join(ev)}\n"
        )
        script.chmod(script.stat().st_mode | stat.S_IXUSR)

    print(f"wrote {len(jobs)} jobs to {jobs_dir.relative_to(repo)}")
    if args.suite == "grid":
        print("NOTE: 36 runs. At ~11 min/run sequential that is ~7h; at -j8 "
              "roughly an hour, less on fast hardware.")


if __name__ == "__main__":
    main()
