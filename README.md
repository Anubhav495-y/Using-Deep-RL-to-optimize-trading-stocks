# Deep Reinforcement Learning for Stock Trading (Finsearch RL)

This is our project for building and training Deep Reinforcement Learning (RL) bots to trade NIFTY 50 constituent stocks (Reliance, TCS, HDFC Bank, and Infosys). 

We built a custom Gymnasium-compatible trading environment that simulates daily trading with realistic transaction fees (0.1%) and price slippage (0.05%). We trained various RL models (DQN, PPO, SAC, TD3) and compared them against traditional baselines like Buy & Hold, ARIMA, and LSTMs using a rolling walk-forward backtest.

---

## Repository Structure

```
project/
  configs/
    ppo_config.yaml          # PPO agent hyperparameters
  data/
    raw/                     # Downloaded raw CSV and Parquet files
    processed/               # Feature datasets
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
      train_agent.py         # Agent training script
      tune_dqn.py            # Optuna tuning script
    evaluation/
      backtester.py          # Deterministic backtesting engine
      run_eval.py            # Backtest evaluation runner
      walk_forward.py        # Walk-forward validation framework
    utils/
      data_downloader.py     # Yahoo Finance historical downloader
  tests/
    test_trading_env.py      # Environment sanity/integration tests
  requirements.txt           # Python dependencies
```

---

## Project Documentation

If you want to read about the different parts of the project, check out these files:

* **[Data Collection](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Data_Collection.md)**: How we downloaded historical data from yfinance and cleaned it.
* **[Feature Engineering](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Feature_Engineering.md)**: The technical indicators we calculated and how we normalized them.
* **[Environment Design](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Environment.md)**: How our custom Gymnasium trading environment handles cash, shares, and transaction fees.
* **[Heuristic Baselines](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Baselines.md)**: Testing simple rules like Buy & Hold, EMA Crossover, and RSI.
* **[Agent Training & Parameters](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Training.md)**: Details on training PPO, DQN, SAC, and TD3.
* **[Evaluation & Backtesting](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Evaluation.md)**: How we backtest our trained models.
* **[Walk-Forward Experiments](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Experiments.md)**: Results from walk-forward testing.
* **[ARIMA & LSTM Benchmarks](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Benchmarks_ARIMA_LSTM.md)**: How statistical and deep learning forecasting models performed.
* **[Final Project Report](file:///mnt/c/Users/Saket/Desktop/Projects/Finsearch_RL/docs/Competition_Winning_Report.md)**: Our final submission report summarizing the progressive features, reward scaling, results, and differentiators.

---

## How to Run the Code

### 1. Environment Setup
Activate your conda environment and install dependencies:
```bash
conda activate RL
pip install -r requirements.txt
```

### 2. Download Raw Data
Download historical daily stock data for the NIFTY 50 index (`^NSEI`) and constituent stocks (`RELIANCE.NS`, `TCS.NS`, `HDFCBANK.NS`, `INFY.NS`):
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

### 5. Train RL Agent
Train the reinforcement learning agent on a stock (e.g. `reliance_ns`):
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

## What We Found (Results Summary)

Here is a summary of our main findings after running rolling backtests from 2021 to 2024:

* **Continuous SAC worked best for Infosys (INFY)**: Our continuous SAC agent achieved a **+108.27% cumulative return** (Sharpe: **0.7676**) using market context features (State 7), easily beating the Buy & Hold benchmark (**+64.56%**).
* **TD3 kept trading costs low**: In the continuous action space, standard SAC traded way too much (over 800 micro-trades) and lost a lot of profit to transaction fees. By using TD3 with deterministic policy smoothing and our downside-deviation Sortino reward, we cut trades to just **19 high-conviction trades** on INFY, yielding **+51.42% return** with a very low drawdown of **-10.68%** (Buy & Hold drawdown was -35.56%).
* **Tuned DQN crushed TCS**: Our tuned DQN agent (optimized with Optuna) achieved the highest return on TCS (**+78.04% return**, Sharpe **0.5914**) using trend indicators (State 2), beating Buy & Hold (**+51.27%**).
* **Adding Market Context helped HDFC Bank**: Moving DQN to State 7 (incorporating rolling Beta and index correlation relative to NIFTY 50) under the downside-only Sortino reward boosted HDFCBANK's return from +37.69% to **+51.73%** (Sharpe: **0.3423**).
* **Forecasting models (ARIMA/LSTM) struggled**: While ARIMA(1,0,1) made some profit, it had high drawdowns. LSTM price forecasting models completely collapsed under noise (near 0% return with high trade counts), showing that predicting price changes directly is much harder than using RL to learn a direct buy/sell policy under trading fees.
* **We optimized the simulator by 30x**: Initially, the environment was very slow because of pandas `.iloc` and `.loc` lookups inside the step loop. By pre-caching the dataframe into flat NumPy arrays, we boosted speed from 2,353 steps/sec to **72,266 steps/second** (a **30.7x speedup**), which allowed us to run parallel Optuna hyperparameter sweeps in minutes.
