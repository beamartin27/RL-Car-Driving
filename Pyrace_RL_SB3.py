"""SB3 runner for Pyrace.

Note: SB3 training (`train`) runs headless by design (no pygame window).
To see the car + radar visualization, use `play` after training.

Commands:
    Train:
        python Pyrace_RL_SB3.py train --env-id Pyrace-v3 --run-name sb3_dqn_v3 --timesteps 200000

        python Pyrace_RL_SB3.py train --env-id Pyrace-v4 --algo ddpg --run-name sb3_ddpg_v4 --timesteps 200000

    Play (renders):
        python Pyrace_RL_SB3.py play --env-id Pyrace-v3 --run-name sb3_dqn_v3

        python Pyrace_RL_SB3.py play --env-id Pyrace-v4 --algo ddpg --run-name sb3_ddpg_v4

Dependency:
    pip install stable-baselines3
"""

from __future__ import annotations

import argparse
import os
import stable_baselines3

import gymnasium as gym

# Registers Pyrace-v1 / Pyrace-v3
import gym_race  # noqa: F401


def make_env(env_id: str, *, seed: int | None, view: bool) -> gym.Env:
    env = gym.make(env_id).unwrapped
    if hasattr(env, "set_view"):
        env.set_view(view)
    env.reset(seed=seed)
    return env


def make_vec_env(env_id: str, *, seed: int | None, view: bool):
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    def _init() -> gym.Env:
        return make_env(env_id, seed=seed, view=view)

    return VecMonitor(DummyVecEnv([_init]))


def cmd_train(env_id: str, algo: str, run_dir: str, timesteps: int, seed: int | None) -> str:
    try:
        from stable_baselines3 import DDPG, DQN
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    os.makedirs(run_dir, exist_ok=True)
    model_stem = os.path.join(run_dir, f"sb3_{algo}_model")

    vec_env = make_vec_env(env_id, seed=seed, view=False)

    if algo == "dqn":
        if not isinstance(vec_env.action_space, gym.spaces.Discrete):
            raise ValueError(f"DQN requires a Discrete action space; got {vec_env.action_space!r}")
        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=seed,
        )
    elif algo == "ddpg":
        if not isinstance(vec_env.action_space, gym.spaces.Box):
            raise ValueError(f"DDPG requires a Box action space; got {vec_env.action_space!r}")
        model = DDPG(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    model.learn(total_timesteps=timesteps)
    model.save(model_stem)
    vec_env.close()

    return model_stem + ".zip"


def cmd_play(env_id: str, algo: str, model_zip_path: str, seed: int | None, max_steps: int) -> None:
    try:
        from stable_baselines3 import DDPG, DQN
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    env = make_env(env_id, seed=seed, view=True)
    if algo == "dqn":
        model = DQN.load(model_zip_path)
    elif algo == "ddpg":
        model = DDPG.load(model_zip_path)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(env.action_space, gym.spaces.Discrete):
            obs, reward, terminated, truncated, _info = env.step(int(action))
        else:
            obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)

        env.render()

        if terminated or truncated:
            break

    env.close()
    print(f"Total reward: {total_reward:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SB3 runner for Pyrace")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train SB3 DQN")
    p_train.add_argument("--env-id", type=str, default="Pyrace-v3")
    p_train.add_argument("--algo", type=str, choices=["dqn", "ddpg"], default="dqn")
    p_train.add_argument("--run-name", type=str, default="sb3_dqn")
    p_train.add_argument("--models-dir", type=str, default="models")
    p_train.add_argument("--timesteps", type=int, default=200_000)
    p_train.add_argument("--seed", type=int, default=0)

    p_play = sub.add_parser("play", help="Play (render) using a trained SB3 DQN")
    p_play.add_argument("--env-id", type=str, default="Pyrace-v3")
    p_play.add_argument("--algo", type=str, choices=["dqn", "ddpg"], default="dqn")
    p_play.add_argument("--run-name", type=str, default="sb3_dqn")
    p_play.add_argument("--models-dir", type=str, default="models")
    p_play.add_argument("--model-path", type=str, default=None)
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--max-steps", type=int, default=2000)

    args = parser.parse_args()
    run_dir = os.path.join(args.models_dir, args.env_id, args.run_name)

    if args.cmd == "train":
        model_zip = cmd_train(args.env_id, args.algo, run_dir, args.timesteps, args.seed)
        print(f"Saved model: {model_zip}")
        return

    if args.cmd == "play":
        model_zip_path = args.model_path
        if model_zip_path is None:
            model_zip_path = os.path.join(run_dir, f"sb3_{args.algo}_model.zip")
        cmd_play(args.env_id, args.algo, model_zip_path, args.seed, args.max_steps)
        return


if __name__ == "__main__":
    main()
