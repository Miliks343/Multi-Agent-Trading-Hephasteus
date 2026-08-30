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

# The grid needs far more seeds than the cheap suite. Measured on the 3-seed
# equalised grid (2026-08-30, M4): the spread of the 12 cell means was sd=838
# while the median within-cell seed sd was 1153, i.e. signal/noise = 0.73 - the
# design could not resolve its own competition effect. Detecting an effect that
# size against that noise at ~2 standard errors needs ~15 seeds per cell.
# Superset of SEEDS, so the 3-seed runs remain a subset.
GRID_SEEDS = tuple(range(15))

# Eval seeds are cheap (~30s each, no training) and a single one leaves every
# number in the writeup resting on one sampled market. Three is the difference
# between "the agent made $X" and "the agent made $X across three markets".
EVAL_SEEDS = (42, 43, 44)


def job(name, train_args, *, n_agents=2, eval_seeds=EVAL_SEEDS, timesteps=50_000):
    return {"name": name, "n_agents": n_agents, "timesteps": timesteps,
            "train_args": list(train_args), "eval_seeds": list(eval_seeds)}


def suite_smoke():
    return [job("smoke", [], timesteps=2_000, eval_seeds=(42,))]


def suite_cheap():
    jobs = []
    # E1 (restated 2026-08-30 - the original premise was wrong). Observations
    # are NOT raw cents: extract_obs already returns prices as distance from
    # mid, sizes over max_size, and clips the whole vector to [-1, 1]. So this
    # is not the reward-scale bug on the input side.
    #
    # What it actually tests: the 44 features sit in [-1, 1] but do not move
    # equally within it. Measured on a rollout, ask_size_L1 has std 0.33 while
    # spread_norm has std 0.00026 - a ~1300x gap. Since first-layer weights all
    # start at the same scale, a feature's influence tracks how much it varies,
    # so the policy is nearly blind to the spread. --norm-obs (VecNormalize
    # per-dim running mean/std) equalises them; the risk is that near-constant
    # features get their jitter amplified to full scale, i.e. noise.
    #
    # Prediction (Pavel, before running): no material difference. Any effect
    # will be smaller than the seed-to-seed variance already known to be large,
    # and the honest reading at 3 seeds will be "this cannot tell us".
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

    Training budget is held constant PER AGENT (2026-08-30). SB3 counts
    total_timesteps as the sum across the vec env, and SuperSuit sets
    num_envs = n_agents, so a flat 50_000 gave each agent 50_176 / 25_088 /
    12_800 env steps at n_agents = 1 / 2 / 4 - a 4x spread in experience and
    in gradient updates, running along the very axis this grid varies. The
    competition effect would have been confounded with training budget.
    Scaling by n_agents gives every agent ~50k env steps regardless.
    """
    jobs = []
    for value_agents in (10, 50, 150, 400):
        for n_agents in (1, 2, 4):
            for s in GRID_SEEDS:
                jobs.append(job(
                    f"v{value_agents}_n{n_agents}/seed{s}",
                    ["--num-value-agents", value_agents, "--seed", s],
                    n_agents=n_agents,
                    timesteps=50_000 * n_agents,
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
        print(f"NOTE: {len(jobs)} runs at {len(GRID_SEEDS)} seeds/cell. "
              "Measured on an M4 at -j8: ~56 min for the 36-job 3-seed "
              "version, so budget ~5h here and far longer on a slow box.")


if __name__ == "__main__":
    main()
