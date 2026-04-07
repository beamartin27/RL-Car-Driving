# Part 02 - Environment Improvements

This section covers the environment-side work for the assignment. The goal is to make the learning problem less sparse and more informative for the agent, without changing the core track or collision logic.

## What was added in `Pyrace-v3`

The new environment version is registered as `Pyrace-v3` in `gym_race/__init__.py` and uses the same base `RaceEnv` class with a `version="v3"` configuration.

### 1. Continuous observations

The original environment discretized each radar reading into integers from 0 to 10. That loses a lot of information because two states that are physically different can collapse into the same bucket.

For `Pyrace-v3`, the observation returned by `observe()` is the raw radar distance for each of the five sensors. This keeps the state continuous and gives function approximators, such as neural networks, much more useful input.

Why this helps:

- The agent can distinguish small changes in track position.
- The learning algorithm does not need to recover information that was removed by rounding.
- It becomes easier to generalize to unseen states.

### 2. Expanded action space with brake

The original setup had only three actions: accelerate, turn left, and turn right. That makes the car hard to control because speed is only reduced passively by friction.

`Pyrace-v3` adds a fourth action:

- `0`: accelerate
- `1`: turn left
- `2`: turn right
- `3`: brake

Why this helps:

- The agent can slow down before tight corners.
- It can recover from overshooting instead of relying only on friction.
- The policy becomes more realistic and easier to learn.

### 3. Better reward shaping

The original reward function was very sparse: the agent mostly received zero reward, a large negative value on crash, and a large positive value only when finishing a lap. That is difficult for exploration-based learning because the agent rarely gets informative feedback.

`Pyrace-v3` uses shaped reward terms:

- A small per-step penalty (`-0.1`) to discourage waiting/stalling.
- A small speed bonus (`+0.02 * speed`) so moving is preferred.
- A progress term for moving closer to the next checkpoint (`+0.05 * Δdistance_to_next_checkpoint`), with the per-step delta clipped to `[-20, 20]` to avoid rare spikes dominating the return.
- A checkpoint bonus (`+15`) when passing a checkpoint.
- A terminal crash penalty (about `-300` to `-240`, depending on checkpoints reached).
- A terminal finish bonus (`+1000`) for completing the lap.
 

Why this helps:

- The agent receives feedback before it completes a full lap.
- Reward is denser than the original sparse setup.
- Checkpoint progress gives an intermediate objective instead of waiting for the final goal.

## Implementation notes

- The continuous-observation version keeps the same five radar sensors.
- The brake action only exists in `Pyrace-v3`, so the original `Pyrace-v1` behavior remains available for comparison.
- The environment now supports both the old sparse reward setup and the improved shaped reward setup through a single code path.

## Expected effect on training

If trained with DQN or another value-based method, the new environment is designed to:

- learn faster than the discrete, sparse-reward version,
- reach checkpoints more reliably,
- and reduce the number of episodes where the agent receives almost no useful learning signal.

These are the expected qualitative effects of the environment changes, based on the added continuous observations, expanded action space, and denser reward signal.

## Possible improvements / future work

Even with the current minimal shaped reward, there are environment-side improvements that could make learning more stable or produce cleaner driving behavior.

### 1) Enable richer radar-based shaping (currently commented out)

The codebase already contains additional reward terms (currently commented out) that use the 5 radar distances:

- **Lane-centering / symmetry**: penalize left-right imbalance so the car stays near the road center.
- **Wall proximity penalty**: penalize being very close to walls (especially front/front-diagonal radars).
- **Speed modulation near danger**: penalize going fast when the forward radars indicate an upcoming wall, and mildly encourage speed when the corridor ahead is clear.

Why this could help:

- It provides earlier, denser feedback than checkpoint-only progress.
- It can reduce the “spin in place / survival” behavior by rewarding controlled, safe motion.

Important caveat:

- These terms must be weighted carefully; too-strong shaping can lead to reward hacking (e.g., driving slowly forever because it avoids penalties). A practical approach is to start with very small coefficients and verify with reward-invariant metrics (goal/crash/timeout, checkpoint count, distance).

### 2) Make episode termination more informative

Currently episodes end only on crash or lap completion (or max steps). A future improvement is to add termination conditions for clearly unproductive behavior (e.g., being stuck for many steps), while keeping the rule consistent across versions.

### 3) Normalize/scale observations consistently

Radar distances are in roughly `[0, 200]`. Normalizing inputs (e.g., dividing by `200`) can help neural network training be more stable and comparable across runs.


## What was added in `Pyrace-v4`

`Pyrace-v4` is registered as `Pyrace-v4` in `gym_race/__init__.py` and uses the same `RaceEnv` wrapper with `version="v4"`.

###  Continuous action space (steer + throttle)

Unlike `Pyrace-v1`/`Pyrace-v3` which use discrete actions, `Pyrace-v4` exposes a continuous 2D action:

- `action[0] = steer` in `[-1, 1]` (negative = turn right, positive = turn left)
- `action[1] = throttle` in `[-1, 1]` (negative = brake, positive = accelerate)

Internally, the simulator applies these as scaled increments each step:

- `angle += 5.0 * steer`
- `speed += 2.0 * throttle`

Why this helps:

- Steering is no longer either right or left. The agent can make small corrections.
- Throttle/brake becomes smooth control rather than discrete jumps.
- In principle, this can produce more stable trajectories and reduce overshooting.

## Expected effect on training (v4)

Because the action space is continuous, value-based discrete-action methods like vanilla DQN are not a natural fit. In practice, `Pyrace-v4` is intended for continuous-control algorithms (e.g., DDPG) that learn an actor (policy) producing real-valued actions.

Expected qualitative effects:

- Potentially smoother control compared to discrete steering/braking.
- More challenging optimization/training than the discrete-action versions, depending on hyperparameters and exploration noise.

