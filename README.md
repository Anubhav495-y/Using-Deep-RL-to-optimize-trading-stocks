# Deep Reinforcement Learning for Stock Trading (Finsearch RL)

A rigorous, research-grade quantitative trading framework that designs, trains, and evaluates Deep Reinforcement Learning (RL) agents on historical NIFTY 50 and constituent stock data.

This project is built using a custom, Gymnasium-compatible portfolio environment, implements realistic market friction (transaction costs and price slippage), establishes heuristic baselines, and evaluates policy models using a time-series walk-forward validation framework.

---

## Repository Structure

```
project/
  configs/
    ppo_config.yaml          # PPO agent hyperparameters
  data/
    raw/                     # Downloaded raw CSV and Parquet files
    processed/               # Stationarized and normalized feature datasets
  docs/
    Data_Collection.md       # Stage 2: Data collection and quality checks
    Feature_Engineering.md   # Stage 3: Feature design, formulas, and math
    Environment.md           # Stage 4: Gym environment and execution mechanics
    Baselines.md             # Stage 5: Rule-based strategy benchmarks
    Training.md              # Stage 6-7: PPO model training and scaling bug fix
    Evaluation.md            # Stage 8: Deterministic backtester & evaluation metrics
    Experiments.md           # Stage 9: Walk-forward time-series cross-validation
  results/
    baselines_comparison.csv # Heuristic baselines performance table
    reliance_ns_backtest.png # Backtest performance plot
    reliance_ns_walk_forward.png # Walk-forward test performance plot
  src/
    environment/
      trading_env.py         # Gymnasium-compliant environment
    features/
      feature_engineer.py    # Technical indicator calculation
      process_all.py         # Batch feature processing pipeline
    baselines/
      baseline_strategies.py # Heuristic strategy definitions
      run_baselines.py       # Baseline evaluation runner
    training/
      train_agent.py         # PPO model training script
    evaluation/
      backtester.py          # Deterministic backtesting engine
      run_eval.py            # Backtest evaluation runner
      walk_forward.py        # Walk-forward validation framework
    utils/
      data_downloader.py     # Yahoo Finance historical downloader
  tests/
    test_trading_env.py      # Environment sanity/integration tests
  requirements.txt           # Python dependency specifications
```

---

## Stage-by-Stage Documentation Index

To deeply understand each phase of the project, refer to the corresponding documentation files:

1. **[Data Collection](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Data_Collection.md)**: Design of the yfinance downloading pipeline and data cleaning/validation logic.
2. **[Feature Engineering](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Feature_Engineering.md)**: Mathematical definitions, financial intuition, and lookahead-bias prevention of the 16 technical and market indicators.
3. **[Environment Design](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Environment.md)**: Gymnasium implementation, action space, state representation, and execution of transaction fees and price slippage.
4. **[Heuristic Baselines](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Baselines.md)**: Definitions and comparative analysis of Buy & Hold, Random, EMA Crossover, and RSI strategies.
5. **[Agent Training & Hyperparameters](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Training.md)**: PPO actor-critic details, custom callbacks, and resolving neural network overflow using dimensionless scaling.
6. **[Evaluation & Backtesting](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Evaluation.md)**: Deterministic evaluation rollout, trade log plotting, and underwater drawdown tracking.
7. **[Walk-Forward Experiments](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Experiments.md)**: Rolling out-of-sample time-series cross-validation design, performance results, and empirical research insights.
8. **[ARIMA & LSTM Benchmarks](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Benchmarks_ARIMA_LSTM.md)**: Rolling walk-forward validation design, mathematical formulations, and out-of-sample results of ARIMA and LSTM baselines.
9. **[Quant-RL Equities Trading Report](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Competition_Winning_Report.md)**: Detailed report detailing progressive state formulations, downside risk engineering, baseline comparisons, and differentiators.

---

## Quick Start & Reproduction

### 1. Environment Setup
Activate your conda environment and install dependencies:
```bash
conda activate RL
pip install -r requirements.txt
```

### 2. Download Raw Data
Download historical daily stock data for the NIFTY 50 index (`^NSEI`) and constituent stocks (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`) from 2015 to 2025:
```bash
python src/utils/data_downloader.py --tickers ^NSEI RELIANCE.NS TCS.NS HDFCBANK.NS INFY.NS --start 2015-01-01 --end 2025-12-31
```

### 3. Compute Features
Run the batch processing pipeline to engineer and scale technical and market indicators:
```bash
python src/features/process_all.py
```

### 4. Evaluate Heuristic Baselines
Simulate rule-based strategies inside the trading environment to establish benchmark metrics:
```bash
python src/baselines/run_baselines.py
```

### 5. Train PPO Agent
Train the reinforcement learning agent on a stock (e.g. `reliance_ns`) using the configured hyperparameters:
```bash
python src/training/train_agent.py --stock reliance_ns --timesteps 100000 --reward portfolio_return
```

### 6. Evaluate and Backtest Model
Run the trained model deterministically on the stock's complete history, calculate performance metrics, and generate comparative plots:
```bash
python src/evaluation/run_eval.py --stock reliance_ns --reward portfolio_return
```

### 7. Run Walk-Forward Validation
Execute rolling time-series cross-validation to evaluate model performance on out-of-sample data (2021-2024):
```bash
python src/evaluation/walk_forward.py --stock reliance_ns --timesteps 50000
```

---

## Key Experimental Results

* **INFY Continuous SAC Champion**: Under the `portfolio_return` reward and State 7 (Market Context) feature group, the continuous SAC agent achieved a **+108.27% cumulative return** (Sharpe: **0.7676**), outperforming the Buy & Hold baseline (**+64.56%**) by a wide margin.
* **INFY TD3 Volatility Regularization**: In the continuous action space under `diff_sortino` and State 7, the TD3 agent achieved a **+51.42% return** with a maximum drawdown of only **-10.68%** (compared to B&H's -35.56%) in just **19 trades**, avoiding transaction cost drag.
* **TCS Champion (V2 DQN Tuned)**: Out-of-sample (2021–2024), the tuned DQN model achieved the highest performance on TCS under State 2, yielding **+78.04% cumulative return** and a **0.5914 Sharpe ratio** with a maximum drawdown of only **-15.67%** (compared to Buy & Hold's +51.27% return).
* **HDFC Bank State 7 DQN Champion**: Moving the tuned DQN agent from State 2 to State 7 (Market Context) under `diff_sortino` boosted returns to **+51.73%** (Sharpe: **0.3423**, Max Drawdown: **-20.87%**).
* **Reliance Default Champion**: The default/untuned DQN model remains the champion configuration for Reliance under State 2, yielding **+47.12% return** and cutting drawdown to **-16.64%** (Sharpe: 0.31). 
* **ARIMA & LSTM Forecasting Baselines**: Out-of-sample (2021–2024), ARIMA(1,0,1) generated positive returns (+28% to +35%) but suffered from high drawdowns. LSTM regressors collapsed under noise (yielding near 0% return and high trade counts), demonstrating that learning a direct policy under transaction costs is far superior to forecasting.
* **30.7x Environment Speedup**: Caching DataFrame queries into flat NumPy arrays inside the step loop increased environment execution speed from 2,353 to **72,266 steps/second**, allowing parallelized hyperparameter optimization.


