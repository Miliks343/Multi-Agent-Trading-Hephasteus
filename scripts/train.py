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
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from marl_lob.env import MarlLobEnv


def make_env(num_vec_envs: int, *, inventory_penalty: float, vecnormalize: bool):
    env = MarlLobEnv(
        n_agents=2,
        max_inventory=10_000,
        inventory_penalty=inventory_penalty,
    )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    env = VecMonitor(env)
    if vecnormalize:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
    return env


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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        set_random_seed(args.seed)

    env = make_env(
        args.num_vec_envs,
        inventory_penalty=args.inventory_penalty,
        vecnormalize=not args.no_vecnormalize,
    )
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
