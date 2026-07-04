# Warm-Up: DQN on the Inverted Pendulum

Before building the trading agents, we did a smaller practice problem to get
comfortable with Deep Q-Networks: balancing the classic **Inverted Pendulum**
(`Pendulum-v1` from Gymnasium). This folder holds that warm-up, written from
scratch (no Stable-Baselines3) so we could see every moving part of DQN ourselves.

The theory behind this is in [`../docs/Pendulum_DQN.md`](../docs/Pendulum_DQN.md)
and in Part 2 of the project report.

## The one catch: DQN needs discrete actions

Pendulum asks for a continuous torque (any float in `[-2.0, 2.0]`), but DQN can only
pick from a fixed list of choices. So we chop the torque range into **11 evenly
spaced values** and let the agent choose one of those 11. A thin environment wrapper
does the translation, so the agent thinks in integers while the physics still gets a
float.

## Files

| File | Phase | What it does |
|------|-------|--------------|
| `discretize_wrapper.py` | 1 | Wraps Pendulum so the agent picks 1 of 11 torque values. |
| `replay_buffer.py`      | 2 | Cyclical memory of past transitions, sampled in random batches. |
| `q_network.py`          | 3 | MLP: state (3) → 128 → 128 → action-values (11). |
| `agent.py`              | 4 | The DQN agent: epsilon-greedy actions, target network, learning step. |
| `train.py`              | 5 | Training loop; saves the model, reward log, and learning curve. |
| `results/`              | — | Trained model, `episode_rewards.csv`, and `learning_curve.png`. |

## How to run

From the repository root, with the project environment active:

```bash
# Full training (500 episodes)
python pendulum_dqn/train.py --episodes 500

# Quick check (fewer episodes)
python pendulum_dqn/train.py --episodes 50 --seed 0
```

Outputs are written to `pendulum_dqn/results/`. Every hyperparameter (learning rate,
number of bins, epsilon schedule, etc.) can be overridden from the command line — run
with `--help` to see them all.

## What "good" looks like

Pendulum rewards are always negative (the penalty per step is smallest when the pole
is upright and still). A random policy scores around **−1000 to −1600** per episode.
Our trained agent (seed 42) climbs to a last-100-episode average of about **−181**,
with the best episodes near **−7** — meaning it has learned to swing the pole up and
hold it near vertical. The moving-average line in `learning_curve.png` shows that climb,
and the full write-up is in [../docs/Pendulum_DQN.md](../docs/Pendulum_DQN.md).
