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

import torch

import gymnasium as gym

# Registers Pyrace-v1 / Pyrace-v3
import gym_race  # noqa: F401


def can_use_progress_bar() -> bool:
    """SB3 progress bar requires optional deps (tqdm + rich)."""
    try:
        import tqdm  # noqa: F401
        import rich  # noqa: F401

        return True
    except Exception:
        return False


def make_env(env_id: str, *, seed: int | None, view: bool) -> gym.Env:
    # Ensure training can run truly headless: avoid opening a Pygame display
    # when `view` is False.
    render_mode = "human" if view else None
    env = gym.make(env_id, render_mode=render_mode).unwrapped
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


def resolve_device(device: str) -> str:
    """Resolve a requested device for SB3.

    - "auto": prefer Apple Silicon GPU (mps), then cuda, else cpu.
    - "mps": Apple Silicon GPU backend (requires torch with MPS support).
    - "cuda": NVIDIA GPU backend (typically unavailable on macOS).
    - "cpu": always CPU.
    """
    device = device.lower().strip()
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested device=cuda but torch.cuda.is_available() is False")
    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        raise RuntimeError("Requested device=mps but torch.backends.mps.is_available() is False")
    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    raise ValueError(f"Unknown device: {device!r}. Use auto|cpu|mps|cuda")


def cmd_train(
    env_id: str,
    algo: str,
    run_dir: str,
    timesteps: int,
    seed: int | None,
    device: str,
    resume_from: str | None,
) -> str:
    try:
        from stable_baselines3 import DDPG, DQN
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    os.makedirs(run_dir, exist_ok=True)
    model_stem = os.path.join(run_dir, f"sb3_{algo}_model")

    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    if resume_from is None:
        print(f"Starting new training run: algo={algo}, timesteps={timesteps}")
    else:
        print(
            f"Resuming training: algo={algo}, timesteps={timesteps}, "
            f"resume_from={resume_from}"
        )

    vec_env = make_vec_env(env_id, seed=seed, view=False)

    # Autosave checkpoints so progress isn't lost if the process is interrupted.
    try:
        from stable_baselines3.common.callbacks import CheckpointCallback
    except Exception:
        CheckpointCallback = None  # type: ignore

    checkpoint_cb = None
    if CheckpointCallback is not None:
        # Save every N environment steps
        checkpoint_cb = CheckpointCallback(
            save_freq=20_000,
            save_path=run_dir,
            name_prefix=f"sb3_{algo}_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

    if resume_from is not None:
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"--resume-from does not exist: {resume_from}")

    if algo == "dqn":
        if not isinstance(vec_env.action_space, gym.spaces.Discrete):
            raise ValueError(f"DQN requires a Discrete action space; got {vec_env.action_space!r}")
        if resume_from is None:
            model = DQN(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                seed=seed,
                device=resolved_device,
            )
        else:
            print(f"Loading DQN from: {resume_from}")
            model = DQN.load(resume_from, env=vec_env, device=resolved_device)
    elif algo == "ddpg":
        if not isinstance(vec_env.action_space, gym.spaces.Box):
            raise ValueError(f"DDPG requires a Box action space; got {vec_env.action_space!r}")
        if resume_from is None:
            model = DDPG(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                seed=seed,
                device=resolved_device,
            )
        else:
            print(f"Loading DDPG from: {resume_from}")
            model = DDPG.load(resume_from, env=vec_env, device=resolved_device)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    use_progress_bar = can_use_progress_bar()
    if not use_progress_bar:
        print("Progress bar disabled (install tqdm+rich to enable it).")

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=checkpoint_cb,
            reset_num_timesteps=(resume_from is None),
            log_interval=1,
            progress_bar=use_progress_bar,
        )
    except ImportError as exc:
        # Some environments report the progress bar deps late or inconsistently.
        # If that happens, fall back to no progress bar instead of crashing.
        msg = str(exc)
        if "install tqdm" in msg or "install tqdm and rich" in msg:
            print(f"Progress bar unavailable ({msg}); retrying without it...")
            model.learn(
                total_timesteps=timesteps,
                callback=checkpoint_cb,
                reset_num_timesteps=(resume_from is None),
                log_interval=1,
                progress_bar=False,
            )
        else:
            raise
    except KeyboardInterrupt:
        interrupted_stem = model_stem + "_interrupted"
        model.save(interrupted_stem)
        vec_env.close()
        print(f"Interrupted; saved: {interrupted_stem}.zip")
        return interrupted_stem + ".zip"

    model.save(model_stem)
    vec_env.close()

    return model_stem + ".zip"


def cmd_play(
    env_id: str,
    algo: str,
    model_zip_path: str,
    seed: int | None,
    max_steps: int,
    device: str,
) -> None:
    try:
        from stable_baselines3 import DDPG, DQN
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "stable_baselines3 is required. Install with: pip install stable-baselines3"
        ) from exc

    env = make_env(env_id, seed=seed, view=True)
    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")
    if algo == "dqn":
        model = DQN.load(model_zip_path, device=resolved_device)
    elif algo == "ddpg":
        model = DDPG.load(model_zip_path, device=resolved_device)
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
    p_train.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Optional path to an existing SB3 .zip model to continue training from.",
    )
    p_train.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device for SB3 (Apple Silicon GPU uses mps).",
    )

    p_play = sub.add_parser("play", help="Play (render) using a trained SB3 DQN")
    p_play.add_argument("--env-id", type=str, default="Pyrace-v3")
    p_play.add_argument("--algo", type=str, choices=["dqn", "ddpg"], default="dqn")
    p_play.add_argument("--run-name", type=str, default="sb3_dqn")
    p_play.add_argument("--models-dir", type=str, default="models")
    p_play.add_argument("--model-path", type=str, default=None)
    p_play.add_argument("--seed", type=int, default=0)
    p_play.add_argument("--max-steps", type=int, default=2000)
    p_play.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device for SB3 during inference.",
    )

    args = parser.parse_args()
    run_dir = os.path.join(args.models_dir, args.env_id, args.run_name)

    if args.cmd == "train":
        model_zip = cmd_train(
            args.env_id,
            args.algo,
            run_dir,
            args.timesteps,
            args.seed,
            args.device,
            args.resume_from,
        )
        print(f"Saved model: {model_zip}")
        return

    if args.cmd == "play":
        model_zip_path = args.model_path
        if model_zip_path is None:
            model_zip_path = os.path.join(run_dir, f"sb3_{args.algo}_model.zip")
        cmd_play(args.env_id, args.algo, model_zip_path, args.seed, args.max_steps, args.device)
        return


if __name__ == "__main__":
    main()
