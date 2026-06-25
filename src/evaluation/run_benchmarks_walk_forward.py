import os
import sys
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.baselines.baseline_strategies import calculate_metrics

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# LSTM model definition
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        last_out = lstm_out[:, -1, :]
        out = self.linear(last_out)
        return out.squeeze(-1)

def run_arima_walk_forward(df_full, windows, stock, feature_group):
    print(f"\n=================== Running ARIMA Walk-Forward for {stock.upper()} ===================")
    current_cash = 100000.0
    out_of_sample_histories = []
    
    # Pre-align indices and get daily return series
    daily_returns = df_full['Daily_Return'].values
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"ARIMA Window {i+1}: Train [{train_start} to {train_end}] | Test [{test_start} to {test_end}]")
        
        train_df = df_full.loc[train_start:train_end]
        test_df = df_full.loc[test_start:test_end]
        
        len_train = len(train_df)
        len_test = len(test_df)
        
        # Get start/end integer indices in df_full
        train_indices = df_full.index.get_indexer(train_df.index)
        test_indices = df_full.index.get_indexer(test_df.index)
        
        start_idx = train_indices[0]
        end_idx = test_indices[-1]
        
        window_returns = df_full.iloc[start_idx:end_idx+1]['Daily_Return'].values
        train_window_returns = window_returns[:len_train]
        
        # Fit ARIMA(1, 0, 1) on train returns
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            warnings.simplefilter('ignore', UserWarning)
            try:
                model = ARIMA(train_window_returns, order=(1, 0, 1))
                res = model.fit()
                res_full = res.apply(window_returns)
                # predict() at index t yields prediction for window_returns[t] using data up to t-1
                # We need forecasts for step k in test period, which corresponds to index (len_train + k)
                # The return of the holding period after step k is window_returns[len_train + k + 1]
                # So the forecast we want is for index len_train + k + 1
                arima_preds = res_full.predict(start=len_train, end=len_train + len_test)
            except Exception as e:
                print(f"ARIMA fitting failed: {e}. Defaulting to zero predictions.")
                arima_preds = np.zeros(len_test + 1)
                
        # Run test simulation
        test_env = TradingEnv(
            df=test_df,
            initial_cash=current_cash,
            transaction_fee=0.001,
            slippage=0.0005,
            reward_type="portfolio_return",
            feature_group=feature_group,
            action_space_type="discrete_3"
        )
        
        obs, info = test_env.reset()
        done = False
        
        while not done:
            k = test_env.current_step
            # Forecast for index len_train + k + 1 corresponds to arima_preds[k + 1] if start was len_train
            if k + 1 < len(arima_preds):
                pred_ret = arima_preds[k + 1]
            else:
                pred_ret = 0.0
                
            # Trading logic
            if pred_ret > 0.0002: # small threshold to prevent noise trading
                action = 1 # Buy
            elif pred_ret < -0.0002:
                action = 2 # Sell
            else:
                action = 0 # Hold
                
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
        history_df = test_env.get_history_df()
        print(f"ARIMA Window {i+1} Finished. Final Portfolio Value: {test_env.portfolio_value:.2f}")
        out_of_sample_histories.append(history_df)
        current_cash = test_env.portfolio_value

    combined_history = pd.concat(out_of_sample_histories, ignore_index=True)
    combined_history['date'] = pd.to_datetime(combined_history['date'])
    combined_history.sort_values('date', inplace=True)
    
    # Calculate metrics
    metrics = calculate_metrics(combined_history, initial_cash=100000.0)
    return metrics, combined_history

def run_lstm_walk_forward(df_full, windows, stock, feature_group, seq_len=10, epochs=40, batch_size=64):
    print(f"\n=================== Running LSTM Walk-Forward for {stock.upper()} ===================")
    current_cash = 100000.0
    out_of_sample_histories = []
    
    # Initialize env just to get correct feature column list
    dummy_env = TradingEnv(df=df_full.iloc[:100], feature_group=feature_group)
    feature_cols = dummy_env.feature_cols
    input_dim = len(feature_cols)
    print(f"LSTM using feature group '{feature_group}' with {input_dim} features.")
    
    # Convert features to numpy
    features_all = df_full[feature_cols].values
    returns_all = df_full['Daily_Return'].values
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LSTM training on device: {device}")
    
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"LSTM Window {i+1}: Train [{train_start} to {train_end}] | Test [{test_start} to {test_end}]")
        
        train_df = df_full.loc[train_start:train_end]
        test_df = df_full.loc[test_start:test_end]
        
        # Get start/end indices in df_full
        train_indices = df_full.index.get_indexer(train_df.index)
        test_indices = df_full.index.get_indexer(test_df.index)
        
        start_idx = train_indices[0]
        len_train = len(train_df)
        
        # 1. Create sequences for training
        # Target for features at index j is returns_all[j+1] (the return of next day)
        X_train, Y_train = [], []
        for j in range(start_idx, start_idx + len_train - seq_len):
            X_train.append(features_all[j : j + seq_len])
            Y_train.append(returns_all[j + seq_len])
            
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
        
        dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize and train LSTM model
        model = LSTMRegressor(input_dim=input_dim, hidden_dim=32, num_layers=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        
        # Run test simulation step-by-step
        test_env = TradingEnv(
            df=test_df,
            initial_cash=current_cash,
            transaction_fee=0.001,
            slippage=0.0005,
            reward_type="portfolio_return",
            feature_group=feature_group,
            action_space_type="discrete_3"
        )
        
        obs, info = test_env.reset()
        done = False
        
        with torch.no_grad():
            while not done:
                k = test_env.current_step
                # The index in the full dataframe corresponding to step k of the test window is:
                idx = test_indices[k]
                
                # To predict return of idx+1, we use features from idx-seq_len+1 to idx
                seq_features = features_all[idx - seq_len + 1 : idx + 1]
                seq_features_tensor = torch.tensor(seq_features, dtype=torch.float32).unsqueeze(0).to(device)
                
                pred_ret = model(seq_features_tensor).item()
                
                # Trading logic
                if pred_ret > 0.0002: # small threshold to prevent noise trading
                    action = 1
                elif pred_ret < -0.0002:
                    action = 2
                else:
                    action = 0
                    
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                
        history_df = test_env.get_history_df()
        print(f"LSTM Window {i+1} Finished. Final Portfolio Value: {test_env.portfolio_value:.2f}")
        out_of_sample_histories.append(history_df)
        current_cash = test_env.portfolio_value

    combined_history = pd.concat(out_of_sample_histories, ignore_index=True)
    combined_history['date'] = pd.to_datetime(combined_history['date'])
    combined_history.sort_values('date', inplace=True)
    
    # Calculate metrics
    metrics = calculate_metrics(combined_history, initial_cash=100000.0)
    return metrics, combined_history

def main():
    parser = argparse.ArgumentParser(description="Run walk-forward benchmarks (ARIMA & LSTM) for tickers.")
    parser.add_argument("--stock", type=str, default="reliance_ns", help="Stock name prefix.")
    parser.add_argument("--feature-group", type=str, default="state_2", help="Feature group for LSTM (default state_2).")
    args = parser.parse_args()
    
    processed_path = f"data/processed/{args.stock}_processed.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data file {processed_path} not found. Run process_all.py first.")
        
    df_full = pd.read_parquet(processed_path)
    df_full.index = pd.to_datetime(df_full.index)
    df_full = df_full.sort_index()
    
    # Define Walk-Forward Windows
    windows = [
        ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
        ("2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
        ("2017-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31")
    ]
    
    # 1. Run ARIMA Benchmark
    arima_metrics, arima_hist = run_arima_walk_forward(df_full, windows, args.stock, args.feature_group)
    
    # 2. Run LSTM Benchmark
    lstm_metrics, lstm_hist = run_lstm_walk_forward(df_full, windows, args.stock, args.feature_group, seq_len=10, epochs=40)
    
    # Save results to CSVs
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    arima_metrics_df = pd.DataFrame([arima_metrics])
    arima_metrics_df["Stock"] = args.stock
    arima_metrics_df["Algorithm"] = "arima"
    arima_metrics_df.to_csv(os.path.join(results_dir, f"{args.stock}_walk_forward_metrics_arima.csv"), index=False)
    
    lstm_metrics_df = pd.DataFrame([lstm_metrics])
    lstm_metrics_df["Stock"] = args.stock
    lstm_metrics_df["Algorithm"] = "lstm"
    lstm_metrics_df.to_csv(os.path.join(results_dir, f"{args.stock}_walk_forward_metrics_lstm.csv"), index=False)
    
    print("\n=================== ARIMA Combined Metrics (2021-2024) ===================")
    for k, v in arima_metrics.items():
        print(f"{k:30} : {v:.4f}" if isinstance(v, float) else f"{k:30} : {v}")
        
    print("\n=================== LSTM Combined Metrics (2021-2024) ===================")
    for k, v in lstm_metrics.items():
        print(f"{k:30} : {v:.4f}" if isinstance(v, float) else f"{k:30} : {v}")
        
    # Plotting comparison
    plt.figure(figsize=(14, 7))
    
    initial_bench_price = df_full.loc["2021-01-01":"2024-12-31"].iloc[0]['Close']
    benchmark_val = 100000.0 * (df_full.loc["2021-01-01":"2024-12-31"]['Close'] / initial_bench_price)
    
    plt.plot(arima_hist['date'], arima_hist['portfolio_value'], label='ARIMA (1,0,1)', color='orange', linewidth=2)
    plt.plot(lstm_hist['date'], lstm_hist['portfolio_value'], label='LSTM Regressor', color='green', linewidth=2)
    plt.plot(benchmark_val.index, benchmark_val.values, label='Buy & Hold Benchmark', color='gray', linestyle='--', linewidth=1.5)
    
    plt.title(f"Walk-Forward Benchmarks comparison: {args.stock.upper()} (2021-2024)", fontsize=14, fontweight='bold')
    plt.ylabel("Portfolio Value (₹)", fontsize=12)
    plt.xlabel("Date", fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plot_path = os.path.join(results_dir, f"{args.stock}_walk_forward_benchmarks.png")
    plt.savefig(plot_path, dpi=300)
    print(f"\nSaved combined benchmarks chart to {plot_path}")
    plt.close()

if __name__ == "__main__":
    main()
