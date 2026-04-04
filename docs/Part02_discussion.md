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

- A small step penalty to discourage doing nothing for too long.
- A small speed bonus so movement is preferable to stalling.
- A progress reward based on getting closer to the next checkpoint.
- A checkpoint bonus when the car passes a checkpoint.
- A larger crash penalty.
- A large finish bonus for completing the lap.

Why this helps:

- The agent receives feedback before it completes a full lap.
- Reward is denser, so learning has a stronger gradient signal.
- Checkpoint progress gives the policy an intermediate objective instead of waiting for the final goal.

## Implementation notes

- The continuous-observation version keeps the same five radar sensors.
- The brake action only exists in `Pyrace-v3`, so the original `Pyrace-v1` behavior remains available for comparison.
- The environment now supports both the old sparse reward setup and the improved shaped reward setup through a single code path.

## Expected effect on training

If trained with DQN or another value-based method, the new environment is designed to:

- learn faster than the discrete, sparse-reward version,
- produce smoother driving behavior,
- reach checkpoints more reliably,
- and reduce the number of episodes where the agent receives almost no useful learning signal.

These are the expected qualitative effects of the environment changes, based on the added continuous observations, expanded action space, and denser reward signal.
