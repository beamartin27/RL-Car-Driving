import sys, os
import math, random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import gym_race
"""
this imports race_env.py (a gym env) and pyrace_2d.py (the race game) and registers the env as "Pyrace-v1"

register(
    id='Pyrace-v1',
    entry_point='gym_race.envs:RaceEnv',
    max_episode_steps=2_000,
)
"""

VERSION_NAME = 'DQN_v01'

REPORT_EPISODES  = 500
DISPLAY_EPISODES = 100


# =============================================================================
# 1. NEURAL NETWORK — replaces the Q-table
# =============================================================================
class DQNetwork(nn.Module):
    """
    Simple MLP that maps state (5 radar values) → Q-values (one per action).
    Architecture: 5 → 64 → 64 → 3
    """
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 2. REPLAY BUFFER — stores transitions for random sampling
# =============================================================================
class ReplayBuffer:
    """
    Fixed-size buffer that stores (state, action, reward, next_state, done) tuples.
    When full, oldest transitions are discarded (deque behavior).
    """
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# =============================================================================
# 3. DQN AGENT — wraps network, buffer, and training logic
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99            # discount factor (same as Q-table version)
        self.lr = 1e-3               # learning rate for Adam optimizer
        self.batch_size = 64         # mini-batch size for replay sampling
        self.min_replay_size = 1000  # don't train until buffer has this many transitions

        # Network and optimizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(capacity=100_000)

    def select_action(self, state, explore_rate):
        """Epsilon-greedy action selection — same logic as Q-table version."""
        if random.random() < explore_rate:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, 5)
                q_values = self.policy_net(state_t)           # shape: (1, 3)
                return q_values.argmax(dim=1).item()          # int

    def train_step(self):
        """
        Sample a mini-batch from replay buffer and perform one gradient update.
        This replaces the single-sample Q-table update from the original code:
            q_table[state_0 + (action,)] += lr * (reward + gamma * best_q - q_table[state_0 + (action,)])
        """
        if len(self.memory) < self.min_replay_size:
            return  # not enough data yet

        # 1. Sample random mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)        # (batch, 5)
        actions_t     = torch.LongTensor(actions).to(self.device)        # (batch,)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)       # (batch,)
        next_states_t = torch.FloatTensor(next_states).to(self.device)   # (batch, 5)
        dones_t       = torch.FloatTensor(dones).to(self.device)         # (batch,)

        # 2. Compute Q_predicted: Q(s, a) for the actions actually taken
        #    policy_net(states_t) gives (batch, 3), we index by the taken action
        q_predicted = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        #    gather(1, ...) selects the Q-value at the action index for each sample

        # 3. Compute Q_target: r + gamma * max Q(s') — with no gradient!
        with torch.no_grad():
            q_next = self.policy_net(next_states_t).max(dim=1)[0]   # max Q-value in next state
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)
            # (1 - dones_t) makes the future term 0 when episode is done

        # 4. Compute loss and backpropagate
        loss = self.loss_fn(q_predicted, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save(self, filepath):
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f'{filepath} saved')

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'{filepath} loaded')


# =============================================================================
# 4. EXPLORATION RATE SCHEDULE — same as Q-table version
# =============================================================================
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


# =============================================================================
# 5. SIMULATE — the main training/playing loop (mirrors Pyrace_RL_QTable.py)
# =============================================================================
def simulate(agent, learning=True, episode_start=0):
    explore_rate = get_explore_rate(episode_start)
    total_reward = 0
    total_rewards = []
    max_reward = -10_000
    threshold = 1000

    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):

        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.plot(total_rewards)
                plt.ylabel('rewards')
                plt.show(block=False)
                plt.pause(4.0)

                # Save memory (for analysis, same as Q-table version)
                file = f'models_{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)

                # Save DQN model (replaces np.save of q_table)
                model_file = f'models_{VERSION_NAME}/model_{episode}.pt'
                agent.save(model_file)

                plt.close()

        obv, _ = env.reset()
        state_0 = np.array(obv, dtype=np.float32)  # use raw observation as float array
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = agent.select_action(state_0, explore_rate if learning else 0)
            obv, reward, done, _, info = env.step(action)
            state = np.array(obv, dtype=np.float32)

            # Store transition in both the env memory (for analysis) and agent replay buffer
            env.remember(tuple(state_0.astype(int)), action, reward, tuple(state.astype(int)), done)
            agent.remember(state_0, action, reward, state, done)
            total_reward += reward

            if learning:
                agent.train_step()  # replaces the Q-table Bellman update

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['DQN SIMULATE',
                              f'Episode: {episode}',
                              f'Time steps: {t}',
                              f'check: {info["check"]}',
                              f'dist: {info["dist"]}',
                              f'crash: {info["crash"]}',
                              f'Reward: {total_reward:.0f}',
                              f'Max Reward: {max_reward:.0f}'])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                break

        explore_rate = get_explore_rate(episode)


# =============================================================================
# 6. LOAD AND PLAY — loads a saved model and plays/continues training
# =============================================================================
def load_and_play(agent, episode, learning=False):
    model_file = f'models_{VERSION_NAME}/model_{episode}.pt'
    agent.load(model_file)
    simulate(agent, learning, episode)


# =============================================================================
# 7. MAIN
# =============================================================================
if __name__ == "__main__":

    env = gym.make("Pyrace-v1").unwrapped
    print('env', type(env))
    if not os.path.exists(f'models_{VERSION_NAME}'):
        os.makedirs(f'models_{VERSION_NAME}')

    STATE_SIZE  = env.observation_space.shape[0]   # 5 (radars)
    ACTION_SIZE = env.action_space.n               # 3 (accelerate, left, right)
    print(f'State size: {STATE_SIZE}, Action size: {ACTION_SIZE}')

    MIN_EXPLORE_RATE = 0.001
    DISCOUNT_FACTOR  = 0.99

    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0
    print(f'Decay factor: {DECAY_FACTOR}')

    NUM_EPISODES = 10_000   # DQN learns faster than Q-table, fewer episodes needed
    MAX_T = 2000

    # Create the DQN agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    print(f'Network: {agent.policy_net}')
    print(f'Device: {agent.device}')

    # -------------
    simulate(agent)  # LEARN from scratch
    # load_and_play(agent, 500, learning=True)   # continue training from saved model
    # load_and_play(agent, 500, learning=False)  # just play with saved model
    # -------------