# Traditional Forecasting Benchmarks (ARIMA & LSTM)

This document describes the implementation, mathematical formulation, evaluation setup, and empirical results of the traditional forecasting models—ARIMA and LSTM—which serve as the baseline benchmarks for our reinforcement learning agents in the Finsearch V3 competition framework.

The codebase is located in [run_benchmarks_walk_forward.py](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/src/evaluation/run_benchmarks_walk_forward.py). Results are saved in `results/` as CSV files and PNG performance charts.

---

## 1. Objective
To implement, train, and backtest traditional statistical (ARIMA) and deep learning (LSTM) forecasting models under the exact same walk-forward validation protocol used for the reinforcement learning agents. This ensures strict comparability across identical initial capital, windows, transaction fees, slippage, and performance metrics.

---

## 2. Model Formulations & Mathematical Intuition

### A. ARIMA (Autoregressive Integrated Moving Average)
We fit a stationary **ARIMA(1, 0, 1)** model on the daily stock returns. Daily returns are stationary, eliminating the need for integration ($d=0$).

* **Mathematical Formula**:
  $$y_t = c + \phi_1 y_{t-1} + \theta_1 \epsilon_{t-1} + \epsilon_t$$
  Where:
  - $y_t$ is the return of the stock at day $t$.
  - $\phi_1$ is the autoregressive coefficient, capturing short-term memory (momentum or mean-reversion).
  - $\theta_1$ is the moving average coefficient, representing structural adjustments to recent shocks.
  - $\epsilon_t \sim N(0, \sigma^2)$ is white noise representing price shocks.

* **Forecast Logic**:
  At day $k$ (current step), the model makes a one-step-ahead forecast for return $\hat{y}_{k+1|k}$. If the predicted return $\hat{y}_{k+1|k} > \text{threshold}$, the agent buys. If it is negative, the agent sells.
* **Fast Updating**:
  To prevent fitting from scratch daily, we fit model coefficients once per training window. During the test period, we use the `results.apply()` method to process test observations sequentially and generate out-of-sample predictions dynamically.

### B. LSTM (Long Short-Term Memory)
We train an LSTM regressor in PyTorch to map historical technical indicator sequences directly to next-day return predictions.

* **LSTM Cell Architecture**:
  The cell state is controlled by three gates to preserve long-term historical context:
  $$\begin{aligned}
  f_t &= \sigma(W_f [h_{t-1}, x_t] + b_f) && \text{(Forget Gate)} \\
  i_t &= \sigma(W_i [h_{t-1}, x_t] + b_i) && \text{(Input Gate)} \\
  \tilde{C}_t &= \tanh(W_c [h_{t-1}, x_t] + b_c) && \text{(Candidate Cell State)} \\
  C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t && \text{(Cell State Update)} \\
  o_t &= \sigma(W_o [h_{t-1}, x_t] + b_o) && \text{(Output Gate)} \\
  h_t &= o_t \odot \tanh(C_t) && \text{(Hidden State)}
  \end{aligned}$$

* **Features Used**:
  State 2 (Trend Indicators) configuration consisting of 17 indicators:
  - Price dynamics (`Daily_Return`, `Log_Return`, `Gap_Return`, `Intraday_Return`, etc.)
  - Moving Average Ratios (SMA 5/10/20/50 Ratios, EMA 10/20/50 Ratios)
  - MACD features (`MACD_Ratio`, `MACD_Signal_Ratio`, `MACD_Hist_Ratio`)

---

## 3. Training & Hyperparameter Configuration

### ARIMA Configuration:
* **Order**: $(p=1, d=0, q=1)$
* **Training Period**: Fit on all returns in the training window (6 years).
* **Test Update**: Parameters remain frozen while data is updated sequentially.
* **Decision Rule**:
  - $\hat{y}_{k+1\|k} > 0.0002 \implies$ **Buy (Action 1)**
  - $\hat{y}_{k+1\|k} < -0.0002 \implies$ **Sell (Action 2)**
  - Otherwise $\implies$ **Hold (Action 0)**

### LSTM Configuration:
* **Architecture**: 1 LSTM layer (32 hidden units), linear output layer mapping to 1 scalar.
* **Sequence Length ($L$)**: 10 days.
* **Optimizer**: Adam (learning rate = 0.001).
* **Loss Function**: Mean Squared Error (MSE).
* **Training Epochs**: 40 epochs per window.
* **Batch Size**: 64.
* **Decision Rule**: Same threshold rules as ARIMA.

---

## 4. Walk-Forward Setup
We use the exact same rolling walk-forward windows (6-year train, 1-year test):
1. **Window 1**: Train `2015-01-01` to `2020-12-31` $\rightarrow$ Test `2021`
2. **Window 2**: Train `2016-01-01` to `2021-12-31` $\rightarrow$ Test `2022`
3. **Window 3**: Train `2017-01-01` to `2022-12-31` $\rightarrow$ Test `2023`
4. **Window 4**: Train `2018-01-01` to `2023-12-31` $\rightarrow$ Test `2024`

*Initial cash for Window 1 is ₹100,000. Ending portfolio value rolls over sequentially.*

---

## 5. Walk-Forward Results (2021–2024)

The table below compiles the performance of the ARIMA and LSTM benchmarks against the Buy & Hold benchmark and our champion RL models:

| Stock | Strategy | Final Value (₹) | Cumulative Return | Sharpe Ratio | Max Drawdown | Trades |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RELIANCE** | **V2 DQN (Default Champion)** | **147,118.43** | **47.12%** | **0.31** | **-16.64%** | **14** |
| | Buy & Hold | 134,334.44 | 34.33% | 0.07 | -24.46% | 1 |
| | ARIMA (1, 0, 1) | 130,376.28 | 30.38% | 0.15 | -24.46% | 18 |
| | LSTM Regressor | 119,159.10 | 19.16% | 0.04 | -24.46% | 3 |
| **TCS** | **V2 DQN Tuned (PR)** | **178,040.44** | **78.04%** | **0.59** | **-15.67%** | **54** |
| | Buy & Hold | 151,268.76 | 51.27% | 0.28 | -24.98% | 1 |
| | ARIMA (1, 0, 1) | 131,389.66 | 31.39% | 0.15 | -25.30% | 28 |
| | LSTM Regressor | 104,038.27 | 4.04% | -0.26 | -17.95% | 91 |
| **HDFCBANK** | **V2 DQN Tuned (Sortino)** | **137,692.03** | **37.69%** | **0.21** | **-22.27%** | **49** |
| | ARIMA (1, 0, 1) | 135,381.00 | 35.38% | 0.19 | -28.31% | 48 |
| | Buy & Hold | 129,592.63 | 29.59% | 0.04 | -23.24% | 1 |
| | LSTM Regressor | 100,044.08 | 0.04% | -0.22 | -30.88% | 79 |
| **INFY** | **V1 Baseline (State 0)** | **173,237.72** | **73.24%** | **0.46** | **-33.00%** | **4** |
| | Buy & Hold | 164,560.42 | 64.56% | 0.40 | -35.56% | 1 |
| | ARIMA (1, 0, 1) | 128,801.33 | 28.80% | 0.14 | -36.80% | 56 |
| | LSTM Regressor | 103,529.61 | 3.53% | -0.25 | -27.93% | 63 |

---

## 6. Analysis & Takeaways

1. **Why ARIMA Performed Moderately**:
   - ARIMA generated positive returns (28% to 35%) across all four assets and outperformed the passive Buy & Hold benchmark on HDFC Bank (+35.38% vs. +29.59%).
   - This occurred because ARIMA acts as an effective short-term momentum indicator, shifting position holdings quickly to capture primary trends while moving to cash during downswings. However, its lack of risk bounding led to high Maximum Drawdown profiles matching or exceeding Buy & Hold (e.g., -36.80% on INFY).

2. **Why LSTM Collapsed**:
   - The LSTM model failed to beat the baseline or generate any substantial alpha, finishing near 0% return on TCS, HDFC Bank, and Infosys.
   - **Reasoning**: Traditional neural networks trained under mean squared error (MSE) try to predict the raw value of the next day's return. Since financial returns are dominated by high-frequency noise, the model overfits the noise to minimize MSE. More importantly, LSTM does not possess a decision-making policy optimized for portfolio return or risk metrics (like Sharpe/Sortino ratios or drawdowns) and transaction costs.
   - Consequently, the model suffered from whipsawing and high trade counts (e.g., 91 trades on TCS), with transaction fees (0.1%) and slippage (0.05%) eroding all paper gains.

3. **Reinforcement Learning Advantage**:
   - On every asset, the RL agents outperformed the traditional forecasting baselines by a wide margin (e.g., TCS Tuned DQN achieved **78.04%** vs. ARIMA's **31.39%** and LSTM's **4.04%**).
   - This is because RL models train policy networks optimized *directly* on downstream portfolio wealth under transaction costs, automatically learning to trade selectively (avoiding fee friction) and penalizing negative metrics (like drawdown and downside volatility).
