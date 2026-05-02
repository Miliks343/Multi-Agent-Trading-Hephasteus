"""Smoke-train PPO on a PettingZoo toy env (simple_spread).

Stand-in for MarlLobEnv. Same wiring (parallel PZ env -> SuperSuit vec env ->
SB3 PPO with parameter sharing) so the swap to MarlLobEnv is just changing the
env import.
"""

import argparse
from pathlib import Path

import supersuit as ss
from mpe2 import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor


def make_env(num_vec_envs: int):
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=True)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=num_vec_envs, num_cpus=1, base_class="stable_baselines3"
    )
    return VecMonitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50_000)
    parser.add_argument("--num-vec-envs", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/toy_simple_spread"))
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout length per env. Smaller -> finer-grained TB logs.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.num_vec_envs)
    # NOTE: not passing seed to PPO — SuperSuit's ConcatVecEnv lacks .seed() and SB3 calls it on init
    model = PPO("MlpPolicy", env, verbose=1, n_steps=args.n_steps, tensorboard_log=str(args.out_dir / "tb"))
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False)
    model.save(args.out_dir / "ppo_simple_spread")

    eval_env = make_env(num_vec_envs=1)
    obs = eval_env.reset()
    total = 0.0
    for _ in range(25):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        total += float(reward.sum())
    print(f"eval episode total reward (sum across agents): {total:.2f}")


if __name__ == "__main__":
    main()
