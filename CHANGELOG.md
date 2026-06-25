# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] - 2026-06-25

### Added
- **Traditional Forecasting Benchmarks (Phase 5A)**:
  - Created a rolling walk-forward evaluation script `src/evaluation/run_benchmarks_walk_forward.py` to train and backtest ARIMA and LSTM models under the exact same transaction costs (0.1%), slippage (0.05%), initial capital (₹100,000), and windows as our RL agents.
  - Implemented an **ARIMA(1, 0, 1)** baseline model using `statsmodels` to forecast daily price returns and convert predictions into Hold/Buy/Sell actions.
  - Implemented a **PyTorch LSTM Regressor** with a 10-day lookback sequence using `torch` to predict returns based on State 2 (Trend) features.
  - Generated out-of-sample walk-forward results, comparison tables, and performance charts for all 4 tickers (`RELIANCE`, `TCS`, `HDFCBANK`, `INFY`).
  - Added new benchmark documentation in `docs/Benchmarks_ARIMA_LSTM.md` detailing formulations, parameters, and findings.
- **Environment Dependencies**: Added `statsmodels` and `patsy` to `requirements.txt`.

### Changed
- **Master Evaluation Tables**: Integrated ARIMA and LSTM walk-forward metrics into the main leaderboards in `docs/Experiments.md` and `docs/Results.md`.
