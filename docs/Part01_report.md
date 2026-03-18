# Assignment 17.00 — Part 01: Q-Table to DQN Conversion

## 1. Objective

Convert the existing Q-Table-based reinforcement learning agent (`Pyrace_RL_QTable.py`) into a Deep Q-Network (DQN) agent, replacing the tabular Q-value lookup with a neural network function approximator while maintaining compatibility with the existing PyRace Gymnasium environment.

## 2. Approach

### 2.1 Why DQN?

The Q-Table approach stores one value per (state, action) pair in a numpy array of shape `(11, 11, 11, 11, 11, 3)` — 483,153 entries. This works because the environment discretizes 5 radar sensors into integers 0–10. However, this approach does not scale: if we wanted finer-grained continuous observations (Part 02), the table would grow exponentially (e.g., 200 values per sensor would require ~960 billion entries).

A neural network approximates Q(s, a) by generalizing across similar states, eliminating the need for explicit enumeration. The DQN takes a state vector as input and outputs Q-values for all actions simultaneously.

### 2.2 Architecture — Q-Table vs DQN Mapping

| Component | Q-Table | DQN |
|---|---|---|
| Q-value storage | `np.zeros((11,11,11,11,11,3))` | Neural network (5 → hidden → 3) |
| Action selection | `np.argmax(q_table[state])` | Forward pass + argmax |
| Learning update | Single-sample Bellman update on table entry | Mini-batch gradient descent with replay buffer |
| Save/load | `np.save(q_table)` | `torch.save(model.state_dict())` |

### 2.3 Key DQN Components

**Experience Replay Buffer**: Stores (state, action, reward, next_state, done) transitions. Random mini-batches of 64 are sampled for training, breaking the temporal correlation between consecutive samples that would otherwise destabilize learning.

**Bellman Target Computation**: For each sampled transition, the target Q-value is computed as:

```
Q_target = reward + γ * max(Q(s', a'))    if not done
Q_target = reward                          if done
```

The target is detached from the gradient graph (computed with `torch.no_grad()`) and treated as a fixed label for the MSE loss.

**Epsilon-Greedy Exploration**: Identical schedule to the Q-Table version — log-based decay from 0.8, with a hard threshold drop to 0.01 at episode 1000. This ensures sufficient exploration before exploitation.

## 3. Implementation

The main deliverable is `Pyrace_RL_DQN.py`, which mirrors the structure of the original `Pyrace_RL_QTable.py`:

- `DQNetwork` class: PyTorch `nn.Module` with configurable hidden layers
- `ReplayBuffer` class: Fixed-size deque storing transitions
- `DQNAgent` class: Wraps network, buffer, and training logic
- `simulate()` function: Same training loop as Q-Table version, with the Q-table Bellman update replaced by `agent.train_step()`
- `load_and_play()` function: Loads saved model weights for evaluation or continued training

No changes were made to the environment files (`gym_race/envs/race_env.py`, `pyrace_2d.py`) for Part 01.

## 4. Ablation Study

To rigorously evaluate which DQN design choices matter most, we conducted an ablation study testing 6 variants. Each changes exactly **one** hyperparameter or architectural choice from the baseline, isolating its effect. All variants use the same training loop, epsilon schedule, and 3,000 episodes.

The experiment configurations were informed by three reference implementations:
- SimoManni/Self-Driving-Car-using-Deep-Q-Learning (Pygame + DQN with radar sensors)
- andywu0913/OpenAI-GYM-CarRacing-DQN (Keras DQN for CarRacing-v0)
- DQN_model.py reference (PyTorch DQN with Double-Q and target network support)
- Arxiv paper 2402.08780: "Enhanced Deep Q-Learning for 2D Self-Driving Cars"

### 4.1 Variants Tested

| Variant | Change from Baseline | Source |
|---|---|---|
| v1_baseline | 5→64→64→3, MSE, γ=0.99, buffer=100k | — |
| v2_combined | v1 + target net + normalization + Huber + wider net | All combined |
| v3_normalize | Inputs divided by 10 → [0, 1] range | Arxiv paper 2402.08780 |
| v4_gamma095 | Discount factor γ = 0.95 | CarRacingDQNAgent |
| v5_wider_net | Single hidden layer: 5→256→3 | SimoManni Q_learning.py |
| v6_small_buffer | Replay buffer = 5,000 entries | CarRacingDQNAgent |
| v7_huber_loss | SmoothL1 (Huber) loss instead of MSE | DQN_model.py |

### 4.2 Results

| Rank | Variant | Lap Completion Rate (last 1k episodes) | Avg Steps per Lap |
|---|---|---|---|
| 1 | **v5_wider_net** | **75.0%** | 701 |
| 2 | v1_baseline | 70.0% | 821 |
| 3 | v4_gamma095 | 65.0% | 904 |
| 4 | v3_normalize | 45.0% | 862 |
| 5 | v6_small_buffer | 40.0% | 977 |
| 6 | v2_combined | 0.0% | — |
| 7 | v7_huber_loss | 0.0% | — |

### 4.3 Analysis

**v5 (wider shallow network) is the best variant.** The SimoManni-inspired architecture of a single 256-unit hidden layer outperforms the deeper 64→64 baseline. With only 5 inputs and 3 outputs, the state-to-action mapping is simple enough that a wide single layer captures it better than stacked narrow layers. It also completes laps faster (701 vs 821 average steps).

**v7 (Huber loss) completely failed** — zero laps across 3,000 episodes. The Huber loss clips large gradients, but with reward values of -10,000 and +10,000, the MSE's large gradients are actually necessary to push Q-values fast enough. Huber made learning too slow to converge within the training budget.

**v2 (combined improvements) also failed.** This is a critical finding: combining multiple "improvements" (target network, normalization, Huber loss, wider network) produced worse results than any single change. The target network, designed to stabilize training by freezing Bellman targets, is counterproductive with this environment's sparse reward structure. The policy network learns from long periods of zero-reward driving followed by rare terminal signals; freezing the targets prevents this information from propagating quickly enough.

**v3 (normalization) hurt performance.** The raw integer inputs (0–10) already provide a good scale for the network. Compressing to [0, 1] may reduce the model's ability to distinguish between adjacent states.

**v6 (small buffer) hurt performance.** With only 5,000 entries, the buffer fills quickly and old experiences are discarded. The agent loses diversity in training data and overfits to recent (often negative) experiences.

**v4 (gamma 0.95) was neutral.** With sparse terminal rewards, the discount factor has minimal impact — the only rewards come at episode end, so intermediate discounting doesn't affect the learning signal substantially.

## 5. Training Dynamics

All successful variants show the same characteristic learning curve:

- **Episodes 0–999 (ε = 0.80)**: Exploration phase. Rewards are consistently between -9,000 and -10,000 (immediate crashes). The replay buffer fills with diverse crash experiences.
- **Episode 1000 (ε drops to 0.01)**: Exploitation begins. The agent immediately starts using the policy learned from 1,000 episodes of crash data. Rewards jump dramatically.
- **Episodes 1000–3000**: Steady improvement as the agent refines its policy through continued training on increasingly successful trajectories.

The sharp transition at episode 1000 demonstrates that the DQN successfully learned useful Q-values during the exploration phase — it just couldn't express them while taking 80% random actions.

## 6. Limitations and Motivation for Part 02

The main bottleneck is the **sparse reward function**: the agent receives 0 reward during normal driving, -10,000 + distance on crash, and +10,000 on lap completion. This means:
- No feedback on driving quality during an episode
- No signal about checkpoint progress
- No reward for speed, efficiency, or safe driving

Even the best variant (v5) still crashes ~25% of the time, likely at specific track sections where the sparse reward provides no guidance. Part 02 addresses this through reward shaping, expanded action spaces (BRAKE), and potentially continuous observations.

## 7. Files

| File | Description |
|---|---|
| `Pyrace_RL_DQN.py` | Main DQN implementation (v1 baseline) |
| `Pyrace_RL_DQN_v2.py` | Combined improvements variant (target net + normalization + Huber) |
| `Pyrace_RL_DQN_experiments.py` | Ablation study runner with 6 configurable variants |
| `models_DQN_v01/` | Saved models and reward plots for v1 |
| `models_DQN_v02/` | Saved models and reward plots for v2 |
| `models_DQN_v{3-7}_*/` | Saved models and reward plots for each ablation variant |