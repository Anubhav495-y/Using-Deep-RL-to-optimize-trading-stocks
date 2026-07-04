"""
Phase 1: Action-space discretization wrapper.

The native Pendulum-v1 environment expects a continuous torque, a float in the
range [-2.0, 2.0]. DQN can only choose from a fixed menu of actions, so we cannot
use Pendulum directly. This wrapper sits between the agent and the environment:
the agent picks an integer 0..10, and the wrapper converts it into the matching
continuous torque before the physics step runs.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    Turns Pendulum-v1's continuous action into a discrete choice of `num_bins`
    evenly spaced torque values.

    Example with num_bins=11: the agent chooses an index 0..10, which maps to
    one of [-2.0, -1.6, -1.2, ..., 1.6, 2.0].
    """

    def __init__(self, env: gym.Env, num_bins: int = 11):
        super().__init__(env)

        # Read the low/high torque limits straight from the wrapped environment
        # so this wrapper still works if the action bounds ever change.
        low = float(env.action_space.low[0])
        high = float(env.action_space.high[0])

        # The fixed menu of continuous torques the agent can pick from.
        self.bins = np.linspace(low, high, num_bins, dtype=np.float32)

        # Tell the agent it is now choosing from `num_bins` discrete actions.
        self.action_space = spaces.Discrete(num_bins)

    def action(self, act: int) -> np.ndarray:
        """Map a discrete index to its continuous torque, shaped (1,) for Pendulum."""
        torque = self.bins[int(act)]
        return np.array([torque], dtype=np.float32)
