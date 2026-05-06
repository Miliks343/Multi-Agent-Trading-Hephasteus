"""Smoke-train PPO on MarlLobEnv (the real thing).

Adapted from train_toy.py — same wiring (PettingZoo parallel env -> SuperSuit
vec env -> SB3 PPO with parameter sharing) but pointed at MarlLobEnv. The
chunk 7 goal is "PPO cold-starts without NaN", not "PPO learns well".

Notes:
- Default num_vec_envs=1 because each ABIDES sim spawns ~2000+ background
  agents; running 8 in parallel is heavy. Bump for real training runs.
- max_inventory bumped to 10_000 so random/early-policy rollouts don't
  trip the per-agent termination cap (which would shrink the agent set
  and break SuperSuit's fixed-agent assumption).
"""

import argparse
from pathlib import Path

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

from marl_lob.env import MarlLobEnv


def make_env(num_vec_envs: int):
    env = MarlLobEnv(n_agents=2, max_inventory=10_000)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    return VecMonitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-vec-envs", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/marl_lob_ppo"))
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout length per env. ABIDES is slow, so "
                             "smaller rollouts give finer-grained TB logs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.num_vec_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=args.n_steps,
        tensorboard_log=str(args.out_dir / "tb"),
    )
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(args.out_dir / "ppo_marl_lob")
    print(f"saved checkpoint -> {args.out_dir / 'ppo_marl_lob.zip'}")


if __name__ == "__main__":
    main()
