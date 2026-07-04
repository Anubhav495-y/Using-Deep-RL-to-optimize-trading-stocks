"""
Phase 5: Training loop.

Runs the full DQN training on the discretized Pendulum-v1 environment, logs the
reward per episode, saves the trained network, and plots the learning curve.

Run from the repository root, for example:

    python pendulum_dqn/train.py --episodes 500
    python pendulum_dqn/train.py --episodes 50 --seed 0      # quick run
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym

# Add the repository root to the path so `pendulum_dqn.*` imports work when this
# script is run directly (mirrors how the trading scripts under src/ do it).
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pendulum_dqn.discretize_wrapper import DiscretizeActionWrapper
from pendulum_dqn.agent import DQNAgent


def train():
    parser = argparse.ArgumentParser(description="Train a DQN agent on discretized Pendulum-v1.")
    parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes.")
    parser.add_argument("--num-bins", type=int, default=11, help="Number of discrete torque actions.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--buffer-capacity", type=int, default=100_000, help="Replay buffer capacity.")
    parser.add_argument("--target-update-freq", type=int, default=100, help="Steps between target syncs.")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate.")
    parser.add_argument("--epsilon-end", type=float, default=0.01, help="Minimum exploration rate.")
    parser.add_argument("--epsilon-decay-episodes", type=int, default=400,
                        help="Episodes over which epsilon decays linearly to its floor.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--results-dir", type=str, default="pendulum_dqn/results", help="Where to save outputs.")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)

    # Build the environment and wrap it so DQN sees a discrete action space.
    env = DiscretizeActionWrapper(gym.make("Pendulum-v1"), num_bins=args.num_bins)
    state_dim = env.observation_space.shape[0]

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=args.num_bins,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
        device=device,
    )

    print("\n=================== Training DQN on Pendulum-v1 (discretized) ===================")
    print(f"Device: {device} | Episodes: {args.episodes} | Discrete actions: {args.num_bins}")

    epsilon = args.epsilon_start
    # Linear decay: subtract a flat amount each episode so epsilon reaches its floor
    # by `epsilon_decay_episodes`. This keeps a rich exploration phase early on and
    # then holds at the floor for the remaining episodes, letting the agent settle
    # cleanly (an exponential schedule can plateau before reaching the floor).
    epsilon_step = (args.epsilon_start - args.epsilon_end) / args.epsilon_decay_episodes

    episode_rewards = []
    recent = []  # rolling window for a smoothed progress read-out

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)

            # The episode ends on either signal, but only `terminated` is a true
            # terminal state. `truncated` is just the 200-step time limit, so we do
            # NOT zero the bootstrap there (Pendulum never actually terminates).
            done = terminated or truncated
            agent.buffer.store(state, action, reward, next_state, terminated)

            state = next_state
            total_reward += reward

            # Learn once per environment step, once enough samples are collected.
            if len(agent.buffer) >= agent.batch_size:
                agent.learn()

        # Decay exploration at the end of each episode, but never below the floor.
        epsilon = max(args.epsilon_end, epsilon - epsilon_step)

        episode_rewards.append(total_reward)
        recent.append(total_reward)
        if len(recent) > 20:
            recent.pop(0)

        if episode % 10 == 0 or episode == 1:
            print(
                f"Episode {episode:4d} | Reward: {total_reward:8.2f} | "
                f"Avg(20): {np.mean(recent):8.2f} | Epsilon: {epsilon:.3f}"
            )

    env.close()

    # Save the trained network and the raw reward log.
    model_path = os.path.join(args.results_dir, "dqn_pendulum.pt")
    agent.save(model_path)
    print(f"\nSaved trained model to {model_path}")

    rewards_path = os.path.join(args.results_dir, "episode_rewards.csv")
    np.savetxt(
        rewards_path,
        np.column_stack([np.arange(1, len(episode_rewards) + 1), episode_rewards]),
        delimiter=",",
        header="episode,reward",
        comments="",
        fmt=["%d", "%.4f"],
    )
    print(f"Saved episode rewards to {rewards_path}")

    plot_learning_curve(episode_rewards, args.results_dir)


def plot_learning_curve(episode_rewards, results_dir):
    """Plot per-episode reward plus a 20-episode moving average."""
    rewards = np.asarray(episode_rewards, dtype=np.float32)
    window = min(20, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(rewards) + 1), rewards,
             color="tab:blue", alpha=0.35, linewidth=1, label="Episode reward")
    plt.plot(np.arange(window, len(rewards) + 1), moving_avg,
             color="tab:blue", linewidth=2.5, label=f"{window}-episode moving average")

    plt.title("DQN on Pendulum-v1: Learning Curve", fontsize=14, fontweight="bold")
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Total Reward", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    plot_path = os.path.join(results_dir, "learning_curve.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved learning curve to {plot_path}")


if __name__ == "__main__":
    train()
