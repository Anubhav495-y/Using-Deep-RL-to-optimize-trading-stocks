"""
Phase 4: The DQN agent.

This ties everything together: it holds the main and target Q-networks, decides how
to act (epsilon-greedy), and performs the learning update. The two DQN stabilising
tricks live here:

  * Target network: a frozen copy of the main network used to compute the training
    target, synced only every `target_update_freq` steps. Without it, the target
    moves every time we update, and training tends to diverge.
  * Experience replay: provided by ReplayBuffer, sampled inside `learn()`.
"""

import numpy as np
import torch
import torch.nn as nn

from pendulum_dqn.q_network import QNetwork
from pendulum_dqn.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 3,
        num_actions: int = 11,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 100,
        device: str = "cpu",
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Main network: trained every step. Target network: a periodically-synced copy.
        self.main_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network = QNetwork(state_dim, num_actions).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(buffer_capacity, state_dim, device)
        self.learn_steps = 0  # counts learning updates, drives the target sync

    def select_action(self, state, epsilon: float) -> int:
        """Epsilon-greedy: explore randomly with prob. epsilon, else act greedily."""
        if np.random.rand() < epsilon:
            return np.random.randint(self.num_actions)

        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.main_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def learn(self):
        """One gradient step on a random mini-batch. No-op until the buffer fills up."""
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Target: reward + gamma * max_a' Q_target(next_state, a'), zeroed at terminal
        # states. Computed under no_grad so no gradient flows through the target network.
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Current estimate: Q(state, action_taken) for the actions actually chosen.
        q_current = self.main_network(states).gather(1, actions)

        loss = self.loss_fn(q_current, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodically copy the main network's weights into the target network.
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())

        return float(loss.item())

    def save(self, path: str):
        torch.save(self.main_network.state_dict(), path)
