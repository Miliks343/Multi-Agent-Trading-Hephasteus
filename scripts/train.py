"""Train PPO on MarlLobEnv (the real thing).

Wiring: PettingZoo parallel env -> SuperSuit vec env -> SB3 PPO with parameter
sharing across the learning agents.

Notes:
- Default num_vec_envs=1 because each ABIDES sim spawns ~2000+ background
  agents; running 8 in parallel is heavy. Bump for real training runs.
- max_inventory bumped to 10_000 so random/early-policy rollouts don't
  trip the per-agent termination cap (which would shrink the agent set
  and break SuperSuit's fixed-agent assumption).
- Defaults are kept exactly as the May 2026 experiments ran them, so results
  in notes/experiments_results.md stay reproducible. In particular
  --inventory-penalty defaults to 0.0 (the env default is 1e-4) and
  observation normalisation is OFF unless --norm-obs is passed.
- device defaults to cpu: the policy is a tiny MLP on a 44-dim observation and
  the bottleneck is the single-threaded ABIDES kernel, so a GPU does not help.
"""

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from marl_lob.env import MarlLobEnv


def make_env(
    num_vec_envs: int,
    *,
    n_agents: int = 2,
    inventory_penalty: float,
    vecnormalize: bool,
    norm_obs: bool = False,
    min_size: int = 0,
    max_offset_cents: int = 50,
    config_kwargs: dict | None = None,
):
    env = MarlLobEnv(
        n_agents=n_agents,
        max_inventory=10_000,
        inventory_penalty=inventory_penalty,
        min_size=min_size,
        max_offset_cents=max_offset_cents,
        config_kwargs=config_kwargs,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    if vecnormalize:
        env = VecNormalize(
            env, norm_obs=norm_obs, norm_reward=True, clip_reward=10.0
        )
    return env


def _git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-vec-envs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/marl_lob_ppo"))
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout length per env. ABIDES is slow, so "
                             "smaller rollouts give finer-grained TB logs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed for PPO + env reset")
    parser.add_argument("--inventory-penalty", type=float, default=0.0,
                        help="passed to MarlLobEnv (env default is 1e-4)")
    parser.add_argument("--no-vecnormalize", action="store_true",
                        help="disable VecNormalize wrapper")

    # --- population / competition ---
    parser.add_argument("--n-agents", type=int, default=2,
                        help="number of learning market makers")

    # --- market composition (forwarded to rmsc03_simple.build_config) ---
    parser.add_argument("--num-value-agents", type=int, default=None,
                        help="informed traders (build_config default 50)")
    parser.add_argument("--num-noise-agents", type=int, default=None,
                        help="uninformed volume (build_config default 2000); "
                             "below ~1000 the L1 barely moves")
    parser.add_argument("--num-momentum-agents", type=int, default=None,
                        help="trend followers (build_config default 10)")

    # --- action-space constraints ---
    parser.add_argument("--min-size", type=int, default=0,
                        help="raise action_space.low on the size dims so "
                             "quoting nothing is impossible")
    parser.add_argument("--max-offset-cents", type=int, default=50,
                        help="cap how far from mid the agent may quote; clamp "
                             "this together with --min-size to close both "
                             "escape hatches at once")

    # --- observation scaling ---
    parser.add_argument("--norm-obs", action="store_true",
                        help="normalise observations in VecNormalize. OFF by "
                             "default to match the May runs, but observations "
                             "are raw cents (~1e5), the same magnitude problem "
                             "that froze the policy via the reward.")

    parser.add_argument("--device", default="cpu",
                        help="torch device; cpu is correct here (tiny MLP, "
                             "simulator-bound)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        set_random_seed(args.seed)

    config_kwargs = {
        k: v
        for k, v in (
            ("num_value_agents", args.num_value_agents),
            ("num_noise_agents", args.num_noise_agents),
            ("num_momentum_agents", args.num_momentum_agents),
        )
        if v is not None
    }

    env = make_env(
        args.num_vec_envs,
        n_agents=args.n_agents,
        inventory_penalty=args.inventory_penalty,
        vecnormalize=not args.no_vecnormalize,
        norm_obs=args.norm_obs,
        min_size=args.min_size,
        max_offset_cents=args.max_offset_cents,
        config_kwargs=config_kwargs or None,
    )
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        tensorboard_log=str(args.out_dir / "tb"),
        device=args.device,
    )

    started = time.time()
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    elapsed = time.time() - started

    model.save(args.out_dir / "ppo_marl_lob")

    # provenance: without this, results collected on different machines at
    # different times cannot be compared or reproduced
    meta = {
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "config_kwargs": config_kwargs,
        "git_sha": _git_sha(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "train_seconds": round(elapsed, 1),
    }
    (args.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"saved checkpoint -> {args.out_dir / 'ppo_marl_lob.zip'}")
    print(f"trained {args.total_timesteps} steps in {elapsed/60:.1f} min "
          f"on {meta['machine']} ({meta['host']})")


if __name__ == "__main__":
    main()
