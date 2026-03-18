"""
Pyrace_RL_DQN_experiments.py — Ablation study of DQN improvements

Run with: python Pyrace_RL_DQN_experiments.py --variant <name>

Variants (each changes ONE thing from v1 baseline):
  v1_baseline    : Original v1 (64→64, MSE, no target net, gamma=0.99)
  v3_normalize   : v1 + input normalization (divide by 10 → [0,1])
  v4_gamma095    : v1 + lower discount factor (0.95 instead of 0.99)
  v5_wider_net   : v1 + wider single hidden layer (5→256→3, SimoManni style)
  v6_small_buffer: v1 + smaller replay buffer (5000 instead of 100,000)
  v7_huber_loss  : v1 + Huber loss (SmoothL1 instead of MSE)

Each variant isolates one technique from the reference implementations:
  - v3: from arxiv paper 2402.08780 (normalized sensor inputs)
  - v4: from CarRacingDQNAgent.py (gamma=0.95)
  - v5: from Q_learning.py / SimoManni (256-unit single hidden layer)
  - v6: from CarRacingDQNAgent.py (memory_size=5000)
  - v7: from DQN_model.py (SmoothL1Loss)
"""

import sys, os, argparse
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
REPORT_EPISODES  = 500
DISPLAY_EPISODES = 100

# =============================================================================
# VARIANT CONFIGURATIONS
# =============================================================================
VARIANTS = {
    'v1_baseline': {
        'hidden_layers': [64, 64],
        'loss': 'mse',
        'gamma': 0.99,
        'buffer_size': 100_000,
        'normalize': False,
        'description': 'Original v1 baseline (for comparison)',
    },
    'v3_normalize': {
        'hidden_layers': [64, 64],
        'loss': 'mse',
        'gamma': 0.99,
        'buffer_size': 100_000,
        'normalize': True,
        'description': 'v1 + input normalization (/10 → [0,1])',
    },
    'v4_gamma095': {
        'hidden_layers': [64, 64],
        'loss': 'mse',
        'gamma': 0.95,
        'buffer_size': 100_000,
        'normalize': False,
        'description': 'v1 + lower gamma (0.95, from CarRacingDQNAgent)',
    },
    'v5_wider_net': {
        'hidden_layers': [256],  # single wide layer, SimoManni style
        'loss': 'mse',
        'gamma': 0.99,
        'buffer_size': 100_000,
        'normalize': False,
        'description': 'v1 + wider shallow net (5→256→3, from SimoManni)',
    },
    'v6_small_buffer': {
        'hidden_layers': [64, 64],
        'loss': 'mse',
        'gamma': 0.99,
        'buffer_size': 5_000,
        'normalize': False,
        'description': 'v1 + small replay buffer (5000, from CarRacingDQN)',
    },
    'v7_huber_loss': {
        'hidden_layers': [64, 64],
        'loss': 'huber',
        'gamma': 0.99,
        'buffer_size': 100_000,
        'normalize': False,
        'description': 'v1 + Huber/SmoothL1 loss (from DQN_model.py)',
    },
}


# =============================================================================
# 1. NEURAL NETWORK — configurable architecture
# =============================================================================
class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super(DQNetwork, self).__init__()
        layers = []
        prev_size = state_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

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
# 3. DQN AGENT — configurable via variant config
# =============================================================================
class DQNAgent:
    def __init__(self, state_size, action_size, config):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config

        # Hyperparameters
        self.gamma = config['gamma']
        self.lr = 1e-3
        self.batch_size = 64
        self.min_replay_size = 1000

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network — architecture from config
        self.policy_net = DQNetwork(state_size, action_size, config['hidden_layers']).to(self.device)
        
        # Loss function — from config
        if config['loss'] == 'huber':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay buffer — size from config
        self.memory = ReplayBuffer(capacity=config['buffer_size'])

    def select_action(self, state, explore_rate):
        if random.random() < explore_rate:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < self.min_replay_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)

        # Q_predicted: same network for both prediction and target (no target net)
        q_predicted = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = self.policy_net(next_states_t).max(dim=1)[0]
            q_target = rewards_t + self.gamma * q_next * (1 - dones_t)

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
# 4. EXPLORATION RATE — same as v1 (log-based + threshold)
# =============================================================================
def get_explore_rate(t):
    return max(MIN_EXPLORE_RATE, min(0.8, 1.0 - math.log10((t + 1) / DECAY_FACTOR)))


# =============================================================================
# 5. SIMULATE
# =============================================================================
def simulate(agent, variant_name, learning=True, episode_start=0):
    explore_rate = get_explore_rate(episode_start)
    total_reward = 0
    total_rewards = []
    max_reward = -10_000
    threshold = 1000
    normalize = agent.config['normalize']

    version_name = f'DQN_{variant_name}'
    if not os.path.exists(f'{MODELS_DIR}/{version_name}'):
        os.makedirs(f'{MODELS_DIR}/{version_name}')

    env.set_view(True)

    for episode in range(episode_start, NUM_EPISODES + episode_start):

        if episode > 0:
            total_rewards.append(total_reward)

            if learning and episode % REPORT_EPISODES == 0:
                plt.figure(figsize=(8, 4))
                plt.plot(total_rewards)
                plt.ylabel('Rewards')
                plt.xlabel('Episode')
                plt.title(f'{variant_name} | Ep {episode} | Max: {max_reward:.0f}')
                plt.tight_layout()
                plt.savefig(f'{MODELS_DIR}/{version_name}/rewards_{episode}.png', dpi=100)
                plt.close()

                model_file = f'{MODELS_DIR}/{version_name}/model_{episode}.pt'
                agent.save(model_file)

                file = f'{MODELS_DIR}/{version_name}/memory_{episode}'
                env.save_memory(file)

        obv, _ = env.reset()
        # Normalize if config says so
        if normalize:
            state_0 = np.array(obv, dtype=np.float32) / 10.0
        else:
            state_0 = np.array(obv, dtype=np.float32)
        total_reward = 0
        if not learning:
            env.pyrace.mode = 2

        if episode >= threshold:
            explore_rate = 0.01

        for t in range(MAX_T):
            action = agent.select_action(state_0, explore_rate if learning else 0)
            obv, reward, done, _, info = env.step(action)
            if normalize:
                state = np.array(obv, dtype=np.float32) / 10.0
            else:
                state = np.array(obv, dtype=np.float32)

            env.remember(tuple(obv), action, reward, tuple(obv), done)
            agent.remember(state_0, action, reward, state, done)
            total_reward += reward

            if learning:
                agent.train_step()

            state_0 = state

            if (episode % DISPLAY_EPISODES == 0) or (env.pyrace.mode == 2):
                env.set_msgs([f'{variant_name}',
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
                if episode % 50 == 0:
                    print(f'Episode {episode}/{NUM_EPISODES + episode_start} | '
                          f'Reward: {total_reward:.0f} | Max: {max_reward:.0f} | '
                          f'Steps: {t} | ε: {explore_rate:.4f}')
                break

        explore_rate = get_explore_rate(episode)

    # Final save
    if learning and total_rewards:
        plt.figure(figsize=(8, 4))
        plt.plot(total_rewards)
        plt.ylabel('Rewards')
        plt.xlabel('Episode')
        plt.title(f'{variant_name} FINAL | {len(total_rewards)} eps | Max: {max_reward:.0f}')
        plt.tight_layout()
        plt.savefig(f'{MODELS_DIR}/{version_name}/rewards_final.png', dpi=100)
        plt.close()
        agent.save(f'{MODELS_DIR}/{version_name}/model_final.pt')

    return total_rewards


# =============================================================================
# 6. MAIN
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN Ablation Experiments')
    parser.add_argument('--variant', type=str, default='v1_baseline',
                        choices=list(VARIANTS.keys()),
                        help='Which variant to run')
    parser.add_argument('--episodes', type=int, default=3000,
                        help='Number of episodes to train')
    args = parser.parse_args()

    variant_name = args.variant
    config = VARIANTS[variant_name]

    print(f'\n{"="*60}')
    print(f'RUNNING VARIANT: {variant_name}')
    print(f'Description: {config["description"]}')
    print(f'Config: {config}')
    print(f'{"="*60}\n')

    env = gym.make("Pyrace-v1").unwrapped
    if not os.path.exists(f'{MODELS_DIR}/DQN_{variant_name}'):
        os.makedirs(f'{MODELS_DIR}/DQN_{variant_name}')

    STATE_SIZE  = env.observation_space.shape[0]
    ACTION_SIZE = env.action_space.n

    MIN_EXPLORE_RATE = 0.001
    NUM_BUCKETS = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    DECAY_FACTOR = np.prod(NUM_BUCKETS, dtype=float) / 10.0

    NUM_EPISODES = args.episodes
    MAX_T = 2000

    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, config)
    print(f'Network: {agent.policy_net}')
    print(f'Loss: {config["loss"]} | Gamma: {config["gamma"]} | Buffer: {config["buffer_size"]} | Normalize: {config["normalize"]}')
    print()

    # Uncomment this for training
    rewards = simulate(agent, variant_name) 

    # Uncomment this for running trained models
    # agent.load(f'{MODELS_DIR}/DQN_{variant_name}/model_final.pt')
    # simulate(agent, variant_name, learning=False, episode_start=3000)