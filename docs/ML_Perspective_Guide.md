# End-to-End Machine Learning Guide: Finsearch RL
This guide explains our reinforcement learning (RL) project using standard machine learning (ML) terminology, bypassing financial jargon. It translates our system's components into data pipelines, state spaces, action bounds, regularized objectives, and validation splits.

---

## 1. The Machine Learning Formulation: Markov Decision Process (MDP)

We formulate the task of determining when to hold, buy, or sell an asset as a **sequential decision-making problem** modeled by a Markov Decision Process (MDP):

```
                 +-------------------+
                 |                   |
                 |     RL Agent      |
                 |  (Neural Network) |
                 |                   |
                 +----+---------^----+
                      |         |
               Action |         | State (Observation)
              (a_t)   |         | (s_t) & Reward (r_t)
                      v         |
                 +----+---------+----+
                 |                   |
                 |    Environment    |
                 |  (State Machine)  |
                 |                   |
                 +-------------------+
```

1. **Agent**: A neural network that learns a policy $\pi(a_t | s_t)$ to map an observation input $s_t$ to a probability distribution over actions $a_t$.
2. **Environment**: A custom state machine that tracks internal variables (e.g., cash, current inventory) and advances one step at a time, calculating state transitions and scalar feedback.
3. **Episode**: A single run through a time-series sequence from start to finish.

---

## 2. The Data & Feature Pipeline

The model receives a sequence of daily feature vectors. The raw price of an asset is **non-stationary** (it grows or falls over time without fixed bounds), which makes model convergence difficult. We transform raw time-series inputs into normalized features.

### Feature Representations
* **State 0 (Baseline)**: Raw scale indicators (raw opening, closing, high, low prices). This leads to scale mismatch and poor performance because values scale upward or downward over years.
* **State 2 (Stationary Trend Features)**: We normalize prices by dividing them by their rolling moving averages:
  $$\text{Feature}_t = \frac{\text{Price}_t}{\text{Rolling Mean}_t(N)}$$
  This bounds features around a stationary midpoint of `1.0`, ensuring the network gets scale-invariant inputs.
* **State 7 (Context Features)**: We append features that measure the correlation, covariance, and trend regime of the individual asset relative to a benchmark index (NIFTY 50). This serves as a global context input, letting the agent know if the broader system is undergoing a regime change.

---

## 3. State, Action, and Reward Design

### A. State (Observation Vector)
At each step $t$, the agent receives a flat vector:
$$S_t = [\text{Inventory Features}, \text{Normalized Technical Features}, \text{Global Context Features}]$$
* **Inventory Features**: Dimensionless state variables representing the agent's current cash and quantity of assets held.
* **Normalized Technical Features**: Ratios, momentum indicator bounds, and volatility metrics (State 2).
* **Global Context Features**: The asset's covariance and beta relative to the index (State 7).

### B. Action Space
We train and evaluate two types of action policies:
1. **Ternary Discrete (`discrete_3`)**: The policy outputs a probability distribution over 3 categorical classes:
   - `0`: Hold current position (do nothing).
   - `1`: Allocate 100% of cash to the asset (buy).
   - `2`: Liquidate 100% of asset holdings to cash (sell).
2. **Continuous Action Space**: The policy outputs a continuous scalar value bounded in $[-1, 1]$:
   - Positive values scale the target holding upward (buying a fraction).
   - Negative values scale the target holding downward (selling a fraction).

### C. Reward Design (Objective Functions)
We use reinforcement learning because it optimizes long-term cumulative reward. We test different feedback signals:
1. **Step Difference (Portfolio Return)**: The scalar difference in the total value of our environment from step $t-1$ to $t$.
2. **Differential Sharpe**: The running mean of step returns divided by the running standard deviation. This penalizes *all* deviation (both upward profits and downward losses).
3. **Differential Sortino**: We optimize downside risk by modifying the variance term to only track **downside deviation** ($B^- = \mathbb{E}[\min(0, x)^2]$). This penalizes negative returns while ignoring positive returns, preventing the agent from liquidating profitable, high-momentum trends prematurely.

---

## 4. Execution Friction as Policy Regularization

In standard control tasks (like CartPole), actions are cost-free. In this environment, executing any action other than "Hold" incurs an **execution fee** (0.1% transaction cost + 0.05% slippage).

$$\text{Transaction Penalty} = 0.3\% \times \text{Volume Traded}$$

This penalty acts as a **policy regularizer**:
* **The Problem**: Without fees, the agent overfits to time-series noise, executing high-frequency micro-trades that collapse under real-world costs (e.g., our stochastic SAC baseline executed 865 trades).
* **The Solution**: Adding the fee directly to step calculations forces the network's gradient update to penalize unnecessary actions. The agent converges on a low-frequency, high-conviction policy that only trades when expected future return exceeds the entry fee.

---

## 5. Walk-Forward Validation (Time-Series Cross-Validation)

Standard cross-validation (like random k-fold) violates temporal order, causing **look-ahead bias** (leaking future data into the past). We use a rolling walk-forward protocol:

```
Window 1:  [ Train: 2015 - 2020 ] -> [ Test: 2021 ]
Window 2:             [ Train: 2016 - 2021 ] -> [ Test: 2022 ]
Window 3:                        [ Train: 2017 - 2022 ] -> [ Test: 2023 ]
Window 4:                                   [ Train: 2018 - 2023 ] -> [ Test: 2024 ]
```

Each window trains a fresh model from scratch on a 6-year history and tests it on a completely unseen 1-year test segment. We then compile the out-of-sample test segments to evaluate generalization.

---

## 6. Supervised Forecasting vs. Reinforcement Learning

It is common to frame time-series trading as a supervised regression task (e.g., using an LSTM to predict tomorrow's price return). Our experiments show why RL is structurally superior:

| Dimension | Supervised Regression (LSTM) | Reinforcement Learning (DQN/SAC) |
| :--- | :--- | :--- |
| **Target Label** | Predicting tomorrow's price change ($y_{t+1} - y_t$). | Maximizing cumulative step rewards ($R_t$) over a rolling horizon. |
| **Loss Function** | Mean Squared Error (MSE) on predictions. | Policy Gradient or Q-value optimization. |
| **Friction Awareness** | Independent of execution fees; requires a secondary heuristic ruleset to make trades. | Fee penalty is directly embedded in the step-reward feedback loop. |
| **Behavioral Result** | Whipsawing and over-trading under noise, causing fee erosion. | Selective, low-frequency trading policies. |

---

## 7. Code Locations for Key Machine Learning Modules

To trace the code logic, inspect these files:

* **Custom Env Interface**: [src/environment/trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py)
  * Study the `reset()` and `step()` functions to see how state transitions and rewards are calculated.
* **Rolling Validation Split**: [src/evaluation/walk_forward.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/walk_forward.py)
  * Trace the rolling window training and backtest rollout loop.
* **Hyperparameter Sweeping**: [src/training/tune_dqn.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/training/tune_dqn.py)
  * See how Optuna runs Bayesian searches over neural network hyperparameter bounds.
* **Supervised Baseline**: [src/evaluation/run_benchmarks_walk_forward.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/run_benchmarks_walk_forward.py)
  * Inspect the LSTM sequence-to-scalar model construction and MSE loss training.
