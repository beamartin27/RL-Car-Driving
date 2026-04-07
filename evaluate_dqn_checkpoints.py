import argparse
import os
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

import gym_race  # noqa: F401 (register envs)


class DQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = state_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class EvalResult:
    episodes: int
    mean_return: float
    mean_steps: float
    goal_rate: float
    crash_rate: float
    timeout_rate: float
    mean_max_check: float
    mean_max_dist: float


def _parse_int_list(csv: str) -> list[int]:
    return [int(x.strip()) for x in csv.split(",") if x.strip()]


def _parse_checkpoints(csv: str) -> list[str]:
    # Allows values like: "500,1000,1500,final"
    return [x.strip() for x in csv.split(",") if x.strip()]


def evaluate_checkpoint(
    env_id: str,
    checkpoint_path: str,
    *,
    hidden_layers: list[int],
    normalize_obs: bool,
    episodes: int,
    max_steps: int,
    seed: int | None,
) -> EvalResult:
    env = gym.make(env_id).unwrapped
    # Disable rendering during evaluation (RaceEnv.reset now preserves this)
    if hasattr(env, "set_view"):
        env.set_view(False)

    state_size = int(env.observation_space.shape[0])
    action_size = int(env.action_space.n)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = DQNetwork(state_size, action_size, hidden_layers).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    # Different scripts saved with different keys; handle both.
    state_dict = ckpt.get("model_state_dict") or ckpt.get("policy_state_dict")
    if state_dict is None:
        raise ValueError(
            f"Unrecognized checkpoint format: {checkpoint_path} (keys={list(ckpt.keys())})"
        )
    net.load_state_dict(state_dict)
    net.eval()

    obs_scale = np.maximum(env.observation_space.high.astype(np.float32), 1.0)

    returns: list[float] = []
    steps_list: list[int] = []
    max_checks: list[int] = []
    max_dists: list[float] = []
    goals = 0
    crashes = 0
    timeouts = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=None if seed is None else seed + ep)
        # Some env implementations reset view flags inside reset(); force headless again.
        if hasattr(env, "set_view"):
            env.set_view(False)
        obs_f = obs.astype(np.float32)
        if normalize_obs:
            obs_f = obs_f / obs_scale

        ep_return = 0.0
        ep_steps = 0
        terminal_goal = False
        terminal_crash = False
        done_reached = False
        ep_max_check = 0
        ep_max_dist = 0.0

        for _t in range(max_steps):
            with torch.no_grad():
                q = net(torch.from_numpy(obs_f).unsqueeze(0).to(device))
                action = int(torch.argmax(q, dim=1).item())

            obs, reward, done, _trunc, info = env.step(action)
            ep_return += float(reward)
            ep_steps += 1

            # Progress signals (available in env info)
            try:
                ep_max_check = max(ep_max_check, int(info.get("check", 0)))
            except Exception:
                pass
            try:
                ep_max_dist = max(ep_max_dist, float(info.get("dist", 0.0)))
            except Exception:
                pass

            obs_f = obs.astype(np.float32)
            if normalize_obs:
                obs_f = obs_f / obs_scale

            if done:
                # Prefer explicit flags from env info when available.
                terminal_crash = bool(info.get("crash", False))
                if "goal" in info:
                    terminal_goal = bool(info.get("goal", False))
                else:
                    # Fallback: in this project, `done` means crash or goal.
                    terminal_goal = not terminal_crash
                done_reached = True
                break

        returns.append(ep_return)
        steps_list.append(ep_steps)
        max_checks.append(ep_max_check)
        max_dists.append(ep_max_dist)
        goals += int(terminal_goal)
        crashes += int(terminal_crash)
        if not done_reached:
            timeouts += 1

    env.close()

    return EvalResult(
        episodes=episodes,
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        goal_rate=goals / episodes if episodes else 0.0,
        crash_rate=crashes / episodes if episodes else 0.0,
        timeout_rate=timeouts / episodes if episodes else 0.0,
        mean_max_check=float(np.mean(max_checks)) if max_checks else 0.0,
        mean_max_dist=float(np.mean(max_dists)) if max_dists else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DQN checkpoints on Pyrace environments (greedy policy)."
    )
    parser.add_argument("--env-id", required=True, help="Gym env id, e.g. Pyrace-v1 or Pyrace-v3")
    parser.add_argument("--run-dir", required=True, help="Directory containing model_<N>.pt files")
    parser.add_argument(
        "--checkpoints",
        default="500,1000,1500,final",
        help="Comma-separated list (supports e.g. 500,1000,1500,final)",
    )
    parser.add_argument(
        "--hidden-layers",
        default="64,64",
        help="Comma-separated hidden layer sizes (must match training)",
    )
    parser.add_argument(
        "--normalize-obs",
        action="store_true",
        help="If set, divides observation by env.observation_space.high (matches v3_normalize training)",
    )
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per checkpoint")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Base seed (deterministic eval)")
    args = parser.parse_args()

    run_dir = args.run_dir
    checkpoints = _parse_checkpoints(args.checkpoints)
    hidden_layers = _parse_int_list(args.hidden_layers)

    print(f"Env: {args.env_id}")
    print(f"Run dir: {run_dir}")
    print(f"Normalize obs: {args.normalize_obs}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Episodes/checkpoint: {args.episodes}\n")

    print(
        "checkpoint\tmean_return\tmean_steps\tgoal_rate\tcrash_rate\t"
        "timeout_rate\tmean_max_check\tmean_max_dist"
    )
    for ck in checkpoints:
        if ck.lower() == "final":
            path = os.path.join(run_dir, "model_final.pt")
            label = "final"
        else:
            path = os.path.join(run_dir, f"model_{ck}.pt")
            label = ck
        if not os.path.exists(path):
            print(f"{label}\tMISSING\t-\t-\t-\t-\t-\t-")
            continue

        res = evaluate_checkpoint(
            args.env_id,
            path,
            hidden_layers=hidden_layers,
            normalize_obs=args.normalize_obs,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        print(
            f"{label}\t{res.mean_return:.1f}\t{res.mean_steps:.1f}\t"
            f"{res.goal_rate:.2f}\t{res.crash_rate:.2f}\t{res.timeout_rate:.2f}\t"
            f"{res.mean_max_check:.2f}\t{res.mean_max_dist:.1f}"
        )


if __name__ == "__main__":
    main()
