# RL-Car-Driving

Gymnasium + Pygame 2D car racing environment with multiple RL baselines:

- Custom DQN implementations (assignment / ablations)
- Stable-Baselines3 baselines (DQN for discrete, DDPG for continuous)
- Checkpoint evaluation + analysis notebooks

## Prerequisites

- macOS / Linux / Windows
- Python 3.10+ (tested with Python 3.12)
- `pip` + a virtualenv recommended

This repo expects you to run commands **from the repository root** so assets like `race_track_ie.png` and `car.png` can be found.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

Install dependencies:

```bash
pip install numpy pandas matplotlib scipy scikit-learn
pip install gymnasium pygame
pip install torch
pip install stable-baselines3
pip install tqdm rich
```

Quick sanity check (imports + env registration):

```bash
python -c "import gymnasium as gym; import gym_race; env=gym.make('Pyrace-v3'); env.reset(); print(env.observation_space, env.action_space); env.close()"
```

## Environments

- `Pyrace-v1`: discrete observation + discrete action
- `Pyrace-v3`: continuous observation + discrete action (shaped rewards)
- `Pyrace-v4`: continuous observation + continuous action (`[steer, throttle]`) for DDPG

## Train (custom DQN)

Available variants are documented at the top of `Pyrace_RL_DQN_experiments.py`.

### Pyrace-v1

```bash
# Ablation variants
python Pyrace_RL_DQN_experiments.py --variant v3_normalize

# Improved DQN v2
python Pyrace_RL_DQN_v2.py --env-id Pyrace-v1 --version-name DQN_v02 --episodes 3000
```

### Pyrace-v3

```bash
# Ablation variants
python Pyrace_RL_DQN_experiments.py --variant v3_normalize --env-id Pyrace-v3

# Improved DQN v2
python Pyrace_RL_DQN_v2.py --env-id Pyrace-v3 --version-name DQN_v02 --episodes 3000
```


## Train / Play (Stable-Baselines3)

SB3 training runs headless by default (no Pygame window). Use `play` to render.

### SB3 DQN (discrete actions)

```bash
python Pyrace_RL_SB3.py train --env-id Pyrace-v3 --algo dqn --run-name sb3_dqn_v3 --timesteps 200000 --device auto
python Pyrace_RL_SB3.py play  --env-id Pyrace-v3 --algo dqn --run-name sb3_dqn_v3
```

### SB3 DDPG (continuous actions)

```bash
python Pyrace_RL_SB3.py train --env-id Pyrace-v4 --algo ddpg --run-name sb3_ddpg_v4 --timesteps 200000 --device auto
python Pyrace_RL_SB3.py play  --env-id Pyrace-v4 --algo ddpg --run-name sb3_ddpg_v4
```

Notes:

- On Apple Silicon, `--device mps` typically gives the best speed.
- If you see an SB3 progress-bar import error, install extras: `pip install 'stable-baselines3[extra]'` (quotes matter in zsh).

## Notebooks / analysis

- `Pyrace_performance_analysis.ipynb`: performance analysis + v1 vs v3 comparison
- `DQN_performance_analysis.ipynb`: additional DQN analysis

Open in VS Code (Jupyter) and run cells with the `.venv` kernel.

## Outputs

Training artifacts are stored under `models/<env-id>/<run-name>/...`.
Examples:

- `models/Pyrace-v3/models_DQN_v3_normalize/model_1000.pt`
- `models/Pyrace-v4/sb3_ddpg_v4/sb3_ddpg_model_*.zip`
