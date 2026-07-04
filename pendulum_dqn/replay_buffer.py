"""
Phase 2: Experience replay buffer.

If we trained on transitions in the exact order they happen, the samples would be
highly correlated (each step looks a lot like the one before it), which makes the
network unstable. The replay buffer fixes this: we store every transition and then
train on a random mini-batch drawn from it. This breaks the correlations and lets
each experience be reused many times.

The buffer is cyclical: once it is full, new transitions overwrite the oldest ones.
"""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size cyclical memory of (state, action, reward, next_state, done)."""

    def __init__(self, capacity: int = 100_000, state_dim: int = 3, device: str = "cpu"):
        self.capacity = capacity
        self.device = device

        # Pre-allocate flat NumPy arrays once, so storing a transition is just an
        # array write (no Python list growth or per-step allocation).
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0        # where the next transition will be written
        self.size = 0       # how many transitions are currently stored

    def store(self, state, action, reward, next_state, done):
        """Insert one transition at the current pointer, wrapping around when full."""
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = 64):
        """Return a random mini-batch as tensors on the configured device."""
        idx = np.random.randint(0, self.size, size=batch_size)
        to_tensor = lambda arr: torch.as_tensor(arr[idx], device=self.device)
        return (
            to_tensor(self.states),
            to_tensor(self.actions),
            to_tensor(self.rewards),
            to_tensor(self.next_states),
            to_tensor(self.dones),
        )

    def __len__(self) -> int:
        return self.size
