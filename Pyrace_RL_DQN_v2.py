"""
Pyrace_RL_DQN_v2.py — Improved DQN for PyRace environment

Improvements over v1 (informed by reference implementations):
  1. Input normalization: radar values divided by 10 → range [0, 1]
     (Arxiv paper 2402.08780 found normalizing sensor values improves DQN)
  2. Target network: a frozen copy updated every N episodes for stable targets
     (Used in CarRacingDQNAgent.py and DQN_model.py references)
  3. Huber loss (SmoothL1): more robust to outlier rewards than MSE
     (Used in DQN_model.py reference)
  4. Multiplicative epsilon decay: smoother exploration schedule
     (Used in all three reference implementations)
  5. Gradient clipping: prevents exploding gradients
  6. Loss tracking: saves loss values for analysis

Reference implementations studied:
  - SimoManni/Self-Driving-Car-using-Deep-Q-Learning (Q_learning.py)
  - andywu0913/OpenAI-GYM-CarRacing-DQN (CarRacingDQNAgent.py)
  - DQN_model.py (full DQN with Double-Q support and TorchRL buffer)
  - Arxiv paper: Enhanced Deep Q-Learning for 2D Self-Driving Cars (2402.08780)
"""

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

MODELS_DIR = 'models'
VERSION_NAME = 'DQN_v02'

REPORT_EPISODES  = 500
DISPLAY_EPISODES = 100


# =============================================================================
# 1. NEURAL NETWORK
# =============================================================================
class DQNetwork(nn.Module):
    """
    MLP: 5 → 128 → 128 → 3
    Wider than v1 (64→64) to give the network more capacity.
    Inspired by Brain class in Q_learning.py (uses 256 hidden units).
    We use 128 as a middle ground — our state space is small (5 inputs).
    """
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)


# =============================================================================
# 2. REPLAY BUFFER
# =============================================================================
class ReplayBuffer:
    def __init__(self, capacity=100_000):
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
# 3. DQN AGENT — with target network and improvements
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # Hyperparameters
        self.gamma = 0.99
        self.lr = 5e-4                 # slightly lower than v1 for stability
        self.batch_size = 64
        self.min_replay_size = 1000

        # Epsilon — multiplicative decay (smoother than log-based)
        # Inspired by CarRacingDQNAgent: epsilon_decay=0.9999
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995    # per-step decay

        # Target network update frequency
        self.target_update_freq = 500  # update target net every N episodes

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (the one we train)
        self.policy_net = DQNetwork(state_size, action_size).to(self.device)
        # Target network (frozen copy, used for stable Bellman targets)
        self.target_net = DQNetwork(state_size, action_size).to(self.device)
        self.update_target_network()  # sync weights at start

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss: more robust than MSE

        # Replay buffer
        self.memory = ReplayBuffer(capacity=100_000)

        # Tracking
        self.losses = []

    def update_target_network(self):
        """Copy policy_net weights into target_net (freeze them there)."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def select_action(self, state, explore_rate=None):
        """Epsilon-greedy. If explore_rate is given, use it; else use self.epsilon."""
        eps = explore_rate if explore_rate is not None else self.epsilon
        if random.random() < eps:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """Sample mini-batch, compute Bellman targets using TARGET network, update policy network."""
        if len(self.memory) < self.min_replay_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q_predicted from POLICY network
        q_predicted = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Q_target from TARGET network (frozen — this is the key improvement)
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

        # Compute loss and backpropagate
        loss = self.loss_fn(q_predicted, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'losses': self.losses[-1000:],  # save last 1000 losses
        }, filepath)
        print(f'{filepath} saved')

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.losses = checkpoint.get('losses', [])
        print(f'{filepath} loaded (epsilon={self.epsilon:.4f})')


# =============================================================================
# 4. STATE NORMALIZATION
# =============================================================================
def normalize_state(obs):
    """
    Normalize observation to [0, 1] range.
    Raw observations are integers 0-10 (radar distance / 20, already discretized).
    Dividing by 10 maps them to [0, 1].

    Why: Neural networks train better with normalized inputs.
    The arxiv paper (2402.08780) normalized their sensor values by dividing
    by max range (1000), finding it helped the DQN learn faster.
    """
    return np.array(obs, dtype=np.float32) / 10.0


# =============================================================================
# 5. SIMULATE
# =============================================================================
def simulate(agent, learning=True, episode_start=0):
    total_reward = 0
    total_rewards = []
    max_reward = -10_000

    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):

        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                # Plot rewards
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(total_rewards)
                plt.ylabel('Rewards')
                plt.xlabel('Episode')
                plt.title(f'Episode {episode} | Max: {max_reward:.0f} | ε: {agent.epsilon:.4f}')

                # Plot loss (if available)
                if agent.losses:
                    plt.subplot(1, 2, 2)
                    # Smooth the loss with a rolling window
                    window = min(100, len(agent.losses))
                    smoothed = np.convolve(agent.losses, np.ones(window)/window, mode='valid')
                    plt.plot(smoothed)
                    plt.ylabel('Loss (smoothed)')
                    plt.xlabel('Training step')
                    plt.title('Training Loss')

                plt.tight_layout()
                plt.savefig(f'{MODELS_DIR}/{VERSION_NAME}/rewards_{episode}.png', dpi=100)
                plt.show(block=False)
                plt.pause(2.0)
                plt.close()

                # Save model
                model_file = f'{MODELS_DIR}/{VERSION_NAME}/model_{episode}.pt'
                agent.save(model_file)

                # Save memory for analysis
                file = f'{MODELS_DIR}/{VERSION_NAME}/memory_{episode}'
                env.save_memory(file)

            # Update target network periodically
            if learning and episode % agent.target_update_freq == 0:
                agent.update_target_network()
                print(f'Episode {episode}: Target network updated | ε={agent.epsilon:.4f}')

        obv, _ = env.reset()
        state_0 = normalize_state(obv)  # NORMALIZED input
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        for t in range(MAX_T):
            # Use agent's internal epsilon (multiplicative decay)
            action = agent.select_action(state_0, explore_rate=0 if not learning else None)
            obv, reward, done, _, info = env.step(action)
            state = normalize_state(obv)  # NORMALIZED input

            # Store in both env memory (analysis) and agent replay buffer
            env.remember(tuple(obv), action, reward, tuple(obv), done)
            agent.remember(state_0, action, reward, state, done)
            total_reward += reward

            if learning:
                agent.train_step()

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs(['DQN v2 SIMULATE',
                              f'Episode: {episode}',
                              f'Time steps: {t}',
                              f'check: {info["check"]}',
                              f'dist: {info["dist"]}',
                              f'crash: {info["crash"]}',
                              f'Reward: {total_reward:.0f}',
                              f'Max Reward: {max_reward:.0f}',
                              f'Epsilon: {agent.epsilon:.4f}'])
                env.render()

            if done or t >= MAX_T - 1:
                if total_reward > max_reward:
                    max_reward = total_reward
                if episode % 50 == 0:
                    print(f'Episode {episode}/{NUM_EPISODES + episode_start} | Reward: {total_reward:.0f} | Max: {max_reward:.0f} | Steps: {t} | ε: {agent.epsilon:.4f}')
                break

    # Final save
    if learning and total_rewards:
        plt.figure()
        plt.plot(total_rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episode')
        plt.title(f'Final — {len(total_rewards)} episodes | Max: {max_reward:.0f}')
        plt.savefig(f'{MODELS_DIR}/{VERSION_NAME}/rewards_final.png', dpi=100)
        plt.close()
        agent.save(f'{MODELS_DIR}/{VERSION_NAME}/model_final.pt')


# =============================================================================
# 6. LOAD AND PLAY
# =============================================================================
def load_and_play(agent, episode, learning=False):
    model_file = f'{MODELS_DIR}/{VERSION_NAME}/model_{episode}.pt'
    agent.load(model_file)
    simulate(agent, learning, episode)


# =============================================================================
# 7. MAIN
# =============================================================================
if __name__ == "__main__":

    env = gym.make("Pyrace-v1").unwrapped
    print('env', type(env))
    if not os.path.exists(f'{MODELS_DIR}/{VERSION_NAME}'):
        os.makedirs(f'{MODELS_DIR}/{VERSION_NAME}')

    STATE_SIZE  = env.observation_space.shape[0]   # 5
    ACTION_SIZE = env.action_space.n               # 3
    print(f'State size: {STATE_SIZE}, Action size: {ACTION_SIZE}')

    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0

    NUM_EPISODES = 3_000
    MAX_T = 2000

    # Create the DQN agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE)
    print(f'Network: {agent.policy_net}')
    print(f'Device: {agent.device}')
    print(f'Improvements: target network, Huber loss, input normalization, gradient clipping')

    # -------------
    simulate(agent)  # LEARN from scratch
    # load_and_play(agent, 500, learning=True)   # continue training
    # load_and_play(agent, 500, learning=False)  # just play
    # -------------