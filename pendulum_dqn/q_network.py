"""
Phase 3: Q-network architecture.

The Q-network replaces the Q-table. It takes the environment state and outputs one
Q-value per discrete action in a single forward pass. The agent then simply picks
the action with the highest predicted value.

For Pendulum the state is 3 numbers (cos angle, sin angle, angular velocity) and we
have 11 discrete torque choices, so this is a small MLP: 3 -> 128 -> 128 -> 11.
"""

import torch.nn as nn


class QNetwork(nn.Module):
    """Multi-layer perceptron mapping a state to action-values."""

    def __init__(self, state_dim: int = 3, num_actions: int = 11, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # No activation on the output: Q-values are unbounded real numbers.
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, state):
        return self.net(state)
