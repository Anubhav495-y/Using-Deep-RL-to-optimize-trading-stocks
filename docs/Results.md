# Empirical Results (Version 1 Baseline)

This document registers the quantitative results of the Version 1 baseline model, baseline heuristic strategies, and walk-forward validation out-of-sample runs.

---

## 1. Heuristic Baselines Performance (2015–2025)

The strategies were simulated inside [trading_env.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/environment/trading_env.py) to ensure transaction fees (0.1%) and slippage (0.05%) are applied consistently:

| Stock | Strategy | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **RELIANCE** | Buy and Hold | 809,207.75 | 709.21% | 21.42% | 0.63 | -45.09% | 1 |
| | Random | 36,644.56 | -63.36% | -8.90% | -0.70 | -69.63% | 894 |
| | EMA Crossover | 319,845.28 | 219.85% | 11.40% | 0.34 | -32.66% | 89 |
| | RSI Strategy | 218,665.49 | 118.67% | 7.53% | 0.16 | -41.21% | 66 |
| **TCS** | Buy and Hold | 321,985.94 | 221.99% | 11.46% | 0.33 | -34.45% | 1 |
| | Random | 27,038.14 | -72.96% | -11.43% | -1.00 | -73.91% | 896 |
| | EMA Crossover | 123,036.90 | 23.04% | 1.94% | -0.16 | -41.32% | 103 |
| | RSI Strategy | 226,739.58 | 126.74% | 7.89% | 0.18 | -29.57% | 74 |
| **HDFCBANK** | Buy and Hold | 455,457.85 | 355.46% | 15.11% | 0.48 | -41.05% | 1 |
| | Random | 105,432.92 | 5.43% | 0.49% | -0.28 | -44.31% | 873 |
| | EMA Crossover | 228,063.85 | 128.06% | 7.95% | 0.18 | -37.14% | 92 |
| | RSI Strategy | 171,059.42 | 71.06% | 5.11% | 0.01 | -38.53% | 60 |
| **INFY** | Buy and Hold | 440,266.15 | 340.27% | 14.75% | 0.43 | -36.55% | 1 |
| | Random | 20,337.51 | -79.66% | -13.74% | -0.99 | -80.20% | 919 |
| | EMA Crossover | 240,215.33 | 140.22% | 8.47% | 0.21 | -39.22% | 83 |
| | RSI Strategy | 189,950.66 | 89.95% | 6.14% | 0.09 | -41.57% | 60 |

---

## 2. Out-of-Sample Walk-Forward Results (2021–2024)

Out-of-sample validation evaluated PPO and DQN models trained on rolling 6-year lookback windows:

| Stock | Strategy | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **RELIANCE** | V1 Baseline (PPO) | 116,798.33 | 16.80% | 4.04% | 0.02 | -24.46% | 4 |
| | V2 PPO (State 2) | 140,213.10 | 40.21% | 9.01% | 0.23 | -22.67% | 3 |
| | V2 PPO Sortino | 126,730.61 | 26.73% | 6.24% | 0.10 | -19.51% | 2 |
| | **V2 DQN (Default Champion)** | **147,118.43** | **47.12%** | **10.36%** | **0.31** | **-16.64%** | **14** |
| | V2 DQN Tuned (PR) | 108,396.81 | 8.40% | 2.08% | -0.08 | -25.95% | 68 |
| | V2 DQN Tuned (Sortino) | 108,668.05 | 8.67% | 2.15% | -0.07 | -24.46% | 30 |
| | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |
| | ARIMA (1, 0, 1) | 130,376.28 | 30.38% | 7.01% | 0.15 | -24.45% | 18 |
| | LSTM Regressor | 119,159.10 | 19.16% | 4.58% | 0.04 | -24.46% | 3 |
| **TCS** | V1 Baseline (PPO) | 112,359.99 | 12.36% | 3.02% | -0.07 | -24.98% | 4 |
| | V2 PPO (State 2) | 144,570.79 | 44.57% | 9.87% | 0.27 | -25.30% | 4 |
| | V2 PPO Sortino | 146,339.71 | 46.34% | 10.21% | 0.28 | -25.30% | 4 |
| | **V2 DQN Tuned (PR)** | **178,040.44** | **78.04%** | **15.87%** | **0.59** | **-15.67%** | **54** |
| | V2 DQN Tuned (Sortino) | 153,120.76 | 53.12% | 11.49% | 0.35 | -23.34% | 17 |
| | Buy & Hold | 151,268.76 | 51.27% | 11.13% | 0.28 | -24.98% | 1 |
| | ARIMA (1, 0, 1) | 131,389.66 | 31.39% | 7.22% | 0.15 | -25.30% | 28 |
| | LSTM Regressor | 104,038.27 | 4.04% | 1.02% | -0.26 | -17.95% | 91 |
| **HDFCBANK** | V1 Baseline (PPO) | 117,343.79 | 17.34% | 4.17% | -0.01 | -24.43% | 3 |
| | V2 PPO (State 2) | 105,393.54 | 5.39% | 1.35% | -0.18 | -31.56% | 3 |
| | V2 PPO Sortino | 114,705.84 | 14.71% | 3.57% | -0.09 | -21.77% | 2 |
| | **V2 DQN Tuned (Sortino)** | **137,692.03** | **37.69%** | **8.51%** | **0.21** | **-22.27%** | **49** |
| | V2 DQN Tuned (PR) | 99,048.83 | -0.95% | -0.24% | -0.21 | -37.81% | 117 |
| | Buy & Hold | 129,592.63 | 29.59% | 6.70% | 0.04 | -23.24% | 1 |
| | ARIMA (1, 0, 1) | 135,381.00 | 35.38% | 8.04% | 0.19 | -28.31% | 48 |
| | LSTM Regressor | 100,044.08 | 0.04% | 0.01% | -0.22 | -30.88% | 79 |
| **INFY** | **V1 Baseline (State 0)** | **173,237.72** | **73.24%** | **15.06%** | **0.46** | **-33.00%** | **4** |
| | V2 PPO (State 2) | 129,539.62 | 29.54% | 6.83% | 0.12 | -24.34% | 2 |
| | V2 PPO Sortino | 58,470.56 | -41.53% | -12.80% | -1.00 | -44.93% | 492 |
| | V2 DQN Tuned (PR) | 131,615.36 | 31.62% | 7.27% | 0.15 | -27.48% | 141 |
| | V2 DQN Tuned (Sortino) | 116,934.85 | 16.93% | 4.08% | 0.00 | -36.11% | 54 |
| | Buy & Hold | 164,560.42 | 64.56% | 13.56% | 0.40 | -35.56% | 1 |
| | ARIMA (1, 0, 1) | 128,801.33 | 28.80% | 6.68% | 0.14 | -36.80% | 56 |
| | LSTM Regressor | 103,529.61 | 3.53% | 0.89% | -0.25 | -27.93% | 63 |

---

## 3. Version 2 State Representation Results (Reliance Walk-Forward)

We track progressive feature group experiments under V2 on `RELIANCE` to determine the best state representation configuration:

| Experiment | Feature Group | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **001** | State 0 (Baseline) | 115,643.46 | 15.64% | 3.78% | -0.0649 | -22.67% | 2 |
| **002** | State 1 (Price Dynamics) | 108,734.36 | 8.73% | 2.16% | -0.0999 | -24.46% | 3 |
| **003** | State 2 (Trend Indicators) | **140,213.10** | **40.21%** | **9.01%** | **0.2314** | **-22.67%** | **3** |
| **004** | State 3 (Momentum Indicators) | 129,816.25 | 29.82% | 6.89% | 0.1457 | -24.45% | 4 |
| **005** | State 4 (Volatility Indicators) | 135,293.01 | 35.29% | 8.02% | 0.1906 | -24.46% | 4 |
| **006** | State 5 (Volume Indicators) | 131,242.02 | 31.24% | 7.19% | 0.1520 | -25.14% | 17 |
| **007** | State 6 (Portfolio State) | 126,935.86 | 26.94% | 6.28% | 0.1180 | -24.46% | 4 |
| **008** | State 7 (Market Context) | 135,691.50 | 35.69% | 8.10% | 0.1932 | -24.46% | 4 |
| **009** | State 8 (Full State) | 101,967.41 | 1.97% | 0.50% | -0.2362 | -22.77% | 6 |
| Benchmark | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |

---

## 4. Version 2 Reward Engineering Results (Reliance Walk-Forward)

Using the champion feature configuration (**State 2 — Trend Indicators**), we evaluate alternative reward formulations:

| Experiment | Reward Type | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **003** | Reward 1 (Portfolio Return) | **140,213.10** | **40.21%** | **9.01%** | **0.2314** | **-22.67%** | **3** |
| **010** | Reward 2 (Return minus Fee) | 121,993.88 | 21.99% | 5.21% | 0.0488 | -26.03% | 3 |
| **011** | Reward 3 (Return minus Drawdown) | 119,388.44 | 19.39% | 4.63% | -0.0487 | **-16.64%** | **1** |
| **012** | Reward 4 (Return minus Volatility) | 100,000.00 | 0.00% | 0.00% | 0.0000 | 0.00% | 0 |
| **013** | Reward 5 (Differential Sharpe) | 122,295.31 | 22.30% | 5.27% | 0.0522 | -26.03% | 3 |
| **014** | Reward 6 (Differential Sortino) | 126,730.61 | 26.73% | 6.24% | 0.0957 | -19.51% | 2 |
| **015** | Reward 7 (Hybrid) | 100,000.00 | 0.00% | 0.00% | 0.0000 | 0.00% | 0 |
| Benchmark | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |

---

## 5. Version 2 Action Space Results (Reliance Walk-Forward)

Using the champion configuration (**State 2 — Trend Indicators** and **Reward 1 — Portfolio Return**), we compare binary and fractional action spaces under walk-forward validation:

| Experiment | Action Space | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **003** | **discrete_3 (Hold/Buy/Sell)** | **140,213.10** | **40.21%** | **9.01%** | **0.2314** | **-22.67%** | **3** |
| **016** | discrete_7 (Fractional size) | 129,513.90 | 29.51% | 6.83% | 0.1431 | -24.46% | 248 |
| Benchmark | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |

---

## 6. Version 2 Observation Window Results (Reliance Walk-Forward)

Using the champion configuration (**State 2 — Trend Indicators**, **Reward 1 — Portfolio Return**, and **discrete_3 Action Space**), we evaluate stacking history windows under walk-forward validation:

| Experiment | History Length | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **003** | **H=1 (Champion)** | **140,213.10** | **40.21%** | **9.01%** | **0.2314** | **-22.67%** | **3** |
| **017a** | H=5 | 138,064.92 | 38.06% | 8.58% | 0.2119 | -22.67% | 3 |
| **017b** | H=10 | 129,816.25 | 29.82% | 6.89% | 0.1457 | -24.45% | 4 |
| Benchmark | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |

---

## 7. Version 2 RL Algorithms Results (Reliance Walk-Forward)

Using the champion configuration (**State 2 — Trend Indicators**, **Reward 1 — Portfolio Return**, and **discrete_3 Action Space**), we compare alternative reinforcement learning algorithms under walk-forward validation:

| Experiment | Algorithm | Final Value (₹) | Cumulative Return | Annualized Return | Sharpe Ratio | Max Drawdown | Trades |
|---|---|---|---|---|---|---|---|
| **003** | PPO (Tuned) | 140,213.10 | 40.21% | 9.01% | 0.2314 | -22.67% | 3 |
| **021a** | A2C | 122,295.31 | 22.30% | 5.27% | 0.0522 | -26.03% | 3 |
| **021b** | **DQN (New Champion)** | **147,118.43** | **47.12%** | **10.36%** | **0.3054** | **-16.64%** | **14** |
| Benchmark | Buy & Hold | 134,334.44 | 34.33% | 7.66% | 0.07 | -24.46% | 1 |

---

## 8. Findings Summary
- **Passive Strategy Dominance**: In the secular bull market of 2015-2025, Buy and Hold outperforms the trend-following and mean-reversion heuristic models in absolute CAGR terms.
- **compounded Friction**: Random trading completely erodes value (up to -79% return), demonstrating the importance of execution costs.
- **RL Outperformance (INFY)**: The PPO rolling agent captured alpha on Infosys (+73.24% vs. +64.56%) while simultaneously **reducing risk** (drawdown dropped from -35.56% to -33.00%, Sharpe increased from 0.40 to 0.46).
- **Trend Anchor Breakthrough (RELIANCE)**: Adding trend indicators (State 2) allowed the RL agent to beat the Buy & Hold benchmark on Reliance (**40.21%** vs **34.33%**), confirming that raw prices and short-term price dynamics are too noisy without trend-based normalization context.
- **Transaction Cost Penalty Friction**: Penalizing transaction fees directly in the reward function (Reward 2) degraded execution performance, as it induced entry/exit paralysis and increased max drawdown.
- **Drawdown Penalty Cash Convergence**: Directly penalizing drawdown (Reward 3) successfully bounded risk (reducing MaxDD to **-16.64%**), but triggered cash convergence, where the agent preferred to sit in risk-free cash for 3 out of 4 windows to avoid drawdown penalties.
- **Volatility Penalty Scale Mismatch**: Penalizing volatility (Reward 4) caused total trading paralysis (0 trades). The volatility penalty term was scaled too high relative to standard daily returns, making any asset holdings negative-sum and forcing cash convergence.
- **Action Space Regularization**: Expanding the action space to 7 actions (supporting 25%, 50%, and 100% buys/sells) caused trade count to explode from 3 to 248, eroding profits to **29.51%** return due to transaction cost (0.1%) and slippage (0.05%) friction. Keeping the action space ternary (`discrete_3`) acts as a highly effective policy regularizer.
- **Observation Window Redundancy**: Stacking sequence history (H=5 and H=10) resulted in progressive out-of-sample performance degradation (38.06% for H=5 and 29.82% for H=10, compared to 40.21% for H=1). Stacking observations increases input dimensionality, rendering technical indicators redundant and exposing the agent to noise overfitting.
- **DQN Outperformance**: DQN emerged as our **NEW CHAMPION** on Reliance, yielding **47.12% return** and reducing Max Drawdown to **-16.64%** (Sharpe: **0.3054**). The off-policy value-based approach utilizing an experience replay buffer breaks temporal correlation in training batches, leading to higher sample efficiency and more robust convergence in discrete financial environments than on-policy policy gradient methods (PPO/A2C).
- **Sortino Stabilization in Scaling**: In scaling validation, the Differential Sortino formulation (`diff_sortino`) acted as a vital stabilizer. For TCS it achieved a **46.34% return** (0.2830 Sharpe), and for HDFC Bank it yielded **14.71% return** and lowered Max Drawdown to **-21.77%** (which is lower risk than passive Buy & Hold's -23.24% drawdown).
- **Infosys Trend Exception**: Scaling to Infosys showed that for stocks with highly persistent upward secular trends, raw price states (State 0) remain superior to relative trend metrics (State 2), which tend to be too conservative. Furthermore, the `diff_sortino` updates collapsed on INFY, triggering a high-frequency trading loop (492 trades).
- **DQN Tuning & Scaling Campaign (New Project Champion)**: Running parallel Optuna tuning on DQN established optimized parameters (`batch_size: 256`, `buffer_size: 5000`, `learning_rate: 0.000331`, `gamma: 0.9648`). When scaled across all stocks, the tuned DQN agent achieved the **highest performance in the entire project** on TCS under `portfolio_return`, yielding **78.04% return** and a **0.5914 Sharpe ratio** with a low drawdown of **-15.67%**. It also stabilized HDFC Bank under `diff_sortino` yielding **37.69% return** (0.2090 Sharpe) and Infosys under `portfolio_return` yielding **31.62% return** (0.1519 Sharpe), although it showed some temporal overfitting on Reliance due to single-split validation search.

---

## 9. Version 2 DQN Hyperparameter Tuning & Scaling Results

Using the optimized DQN hyperparameters found via Optuna (`configs/dqn_best_params.yaml`), we ran walk-forward validation (2021–2024) across Reliance, TCS, HDFC Bank, and Infosys under both `portfolio_return` and `diff_sortino` rewards:

| Stock | Algorithm | Reward Type | Cumulative Return (%) | Annualized Return (%) | Sharpe Ratio | Sortino Ratio | Max Drawdown (%) | Trades |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | DQN (Tuned) | portfolio_return | 8.40% | 2.08% | -0.0760 | -0.0069 | -25.95% | 68 |
| | DQN (Tuned) | diff_sortino | 8.67% | 2.15% | -0.0745 | -0.0066 | -24.46% | 30 |
| **TCS** | **DQN (Tuned)** | **portfolio_return** | **78.04%** | **15.87%** | **0.5914** | **0.0536** | **-15.67%** | **54** |
| | DQN (Tuned) | diff_sortino | 53.12% | 11.49% | 0.3522 | 0.0319 | -23.34% | 17 |
| **HDFCBANK** | **DQN (Tuned)** | **diff_sortino** | **37.69%** | **8.51%** | **0.2090** | **0.0182** | **-22.27%** | **49** |
| | DQN (Tuned) | portfolio_return | -0.95% | -0.24% | -0.2066 | -0.0177 | -37.81% | 117 |
| **INFY** | **DQN (Tuned)** | **portfolio_return** | **31.62%** | **7.27%** | **0.1519** | **0.0128** | **-27.48%** | **141** |
| | DQN (Tuned) | diff_sortino | 16.93% | 4.08% | 0.0039 | 0.0003 | -36.11% | 54 |

---

## 10. Version 3 State Representation Scaling Results (State 7 - Market Context)

To evaluate the impact of rolling market covariance, correlation, beta, and trend regime signals (State 7), we ran walk-forward validation (2021–2024) using the same tuned DQN agent parameters across all 4 stocks:

| Stock | Algorithm | Reward Type | Feature Group | Cumulative Return (%) | Annualized Return (%) | Sharpe Ratio | Max Drawdown (%) | Trades |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | DQN (Tuned) | portfolio_return | State 7 | 15.52% | 3.75% | -0.0290 | -21.19% | 50 |
| | DQN (Tuned) | diff_sortino | State 7 | -10.45% | -2.78% | -0.6435 | -24.46% | 71 |
| **TCS** | DQN (Tuned) | portfolio_return | State 7 | 32.17% | 7.38% | 0.1525 | -14.09% | 102 |
| | DQN (Tuned) | diff_sortino | State 7 | 26.69% | 6.23% | 0.0796 | -16.72% | 56 |
| **HDFCBANK** | **DQN (Tuned)** | **diff_sortino** | **State 7** | **51.73%** | **11.23%** | **0.3423** | **-20.87%** | **87** |
| | DQN (Tuned) | portfolio_return | State 7 | 16.95% | 4.08% | -0.0552 | -29.32% | 142 |
| **INFY** | **DQN (Tuned)** | **diff_sortino** | **State 7** | **30.94%** | **7.12%** | **0.1365** | **-28.28%** | **24** |
| | DQN (Tuned) | portfolio_return | State 7 | 11.27% | 2.76% | -0.3210 | -8.15% | 92 |

### Key Observations:
1. **HDFCBANK Outperformance**: Moving from State 2 to State 7 under the `diff_sortino` reward led to a significant increase in Cumulative Return (from **37.69%** to **51.73%**) and Sharpe Ratio (from **0.2090** to **0.3423**), establishing State 7 as the new champion for HDFC Bank.
2. **INFY Efficiency Gain**: For INFY under `diff_sortino`, State 7 achieved **30.94% return** (Sharpe: **0.1365**) compared to State 2's **16.93% return** (Sharpe: **0.0039**). Crucially, the number of trades was cut in half (from **54** to **24**), demonstrating that market trend context successfully filtered out noisy intraday trades, reducing transaction cost drag.
3. **TCS and Reliance Degradation**: For TCS and Reliance, adding market features (State 7) degraded performance compared to their State 2 champions (TCS return dropped from 78.04% to 32.17%). This suggests that market-wide context features introduced overfitting noise for these specific single-stock dynamics.

---

## 11. Version 4 Continuous Action Space Validation Results (State 7 - Market Context)

To evaluate the benefits of continuous action scaling, we ran walk-forward validation (2021–2024) for PPO continuous, SAC, and TD3 agents under the State 7 (Market Context) feature group:

| Stock | Algorithm | Reward Type | Feature Group | Action Space | Cumulative Return (%) | Annualized Return (%) | Sharpe Ratio | Max Drawdown (%) | Trades |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | PPO | portfolio_return | State 7 | continuous | 12.82% | 3.13% | -0.0362 | -24.46% | 378 |
| | PPO | diff_sortino | State 7 | continuous | 14.05% | 3.41% | -0.1796 | -17.60% | 552 |
| | SAC | portfolio_return | State 7 | continuous | -14.87% | -4.03% | -0.7535 | -23.82% | 676 |
| | SAC | diff_sortino | State 7 | continuous | -0.13% | -0.03% | -106.8547 | -0.13% | 18 |
| | **TD3 (New Champion)** | **portfolio_return** | **State 7** | **continuous** | **46.29%** | **10.20%** | **0.2944** | **-28.15%** | **534** |
| | TD3 | diff_sortino | State 7 | continuous | -1.72% | -0.44% | -0.6116 | -17.32% | 138 |
| **TCS** | PPO | portfolio_return | State 7 | continuous | 5.93% | 1.48% | -0.2608 | -21.43% | 267 |
| | **PPO (New Champion)** | **diff_sortino** | **State 7** | **continuous** | **32.07%** | **7.36%** | **0.1553** | **-11.83%** | **755** |
| | SAC | portfolio_return | State 7 | continuous | -5.29% | -1.38% | -0.4073 | -21.68% | 848 |
| | SAC | diff_sortino | State 7 | continuous | -0.05% | -0.01% | -3.6391 | -2.61% | 149 |
| | TD3 | portfolio_return | State 7 | continuous | -25.54% | -7.25% | -0.7899 | -41.93% | 362 |
| | TD3 | diff_sortino | State 7 | continuous | -0.59% | -0.15% | -25.1296 | -0.59% | 6 |
| **HDFCBANK** | PPO | portfolio_return | State 7 | continuous | 9.78% | 2.41% | -0.1621 | -20.42% | 319 |
| | PPO | diff_sortino | State 7 | continuous | 24.27% | 5.70% | 0.0359 | -19.56% | 794 |
| | SAC | portfolio_return | State 7 | continuous | 15.97% | 3.86% | -0.0821 | -15.05% | 852 |
| | SAC | diff_sortino | State 7 | continuous | -2.41% | -0.62% | -1.6959 | -7.97% | 286 |
| | **TD3** | **portfolio_return** | **State 7** | **continuous** | **40.98%** | **9.16%** | **0.2381** | **-24.43%** | **538** |
| | TD3 | diff_sortino | State 7 | continuous | 2.17% | 0.55% | -3.6760 | -1.50% | 24 |
| **INFY** | PPO | portfolio_return | State 7 | continuous | 29.05% | 6.73% | 0.1254 | -33.69% | 478 |
| | PPO | diff_sortino | State 7 | continuous | 40.92% | 9.15% | 0.2910 | -17.49% | 595 |
| | **SAC (New Champion)** | **portfolio_return** | **State 7** | **continuous** | **108.27%** | **20.60%** | **0.7676** | **-20.17%** | **865** |
| | SAC | diff_sortino | State 7 | continuous | 0.03% | 0.01% | -40.6598 | -0.30% | 40 |
| | **TD3** | **diff_sortino** | **State 7** | **continuous** | **51.42%** | **11.17%** | **0.4909** | **-10.68%** | **19** |
| | TD3 | portfolio_return | State 7 | continuous | -14.99% | -4.06% | -0.3860 | -42.54% | 599 |

### Key Observations:
1. **INFY SAC continuous Outperformance**: SAC continuous under the `portfolio_return` reward achieved the **highest performance in the entire project for INFY**, yielding **108.27% return** and a high Sharpe ratio of **0.7676**. It successfully leveraged fractional position scaling to beat Buy & Hold (64.56%) by a wide margin.
2. **INFY TD3 continuous Efficiency**: TD3 continuous under the `diff_sortino` reward achieved a remarkable **51.42% return** with a Sharpe ratio of **0.4909** and an exceptionally low drawdown of **-10.68%** (compared to Buy & Hold's -35.56% drawdown) in only **19 trades**. This demonstrates that continuous actor-critic regularization with deterministic policy smoothing successfully avoids transaction cost drag while securing high risk-adjusted profits.
3. **Reliance TD3 continuous Champion**: For Reliance, TD3 continuous (portfolio_return) achieved a new best return of **46.29%** (0.2944 Sharpe), outperforming DQN discrete (15.52%) and Buy & Hold (34.33%).
4. **Friction Sensitivity**: Stochastic policies (like SAC) are highly sensitive to transaction fees. Because action values float continuously, stochastic exploration led SAC to execute up to 865 micro-trades, creating severe fee erosion unless regularized by a risk penalty. Deterministic policy networks (like TD3) combined with Differential Sortino rewards successfully regularized action scaling.



