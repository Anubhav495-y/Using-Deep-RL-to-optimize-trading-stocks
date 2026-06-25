import os
import sys
import argparse
import yaml
import pandas as pd
import numpy as np
import optuna
import warnings
from stable_baselines3 import PPO, A2C, DQN

# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.environment.trading_env import TradingEnv
from src.evaluation.backtester import Backtester
from src.baselines.baseline_strategies import calculate_metrics
from src.utils.device_utils import select_device

# Filter warnings
warnings.filterwarnings("ignore")

# Define Walk-Forward Windows
WINDOWS = [
    ("2015-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2016-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2017-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31")
]

class WalkForwardObjective:
    def __init__(self, stock, algo, df_full, feature_group, reward_type, action_space_type, timesteps, device="cpu"):
        self.stock = stock
        self.algo = algo
        self.df_full = df_full
        self.feature_group = feature_group
        self.reward_type = reward_type
        self.action_space_type = action_space_type
        self.timesteps = timesteps
        self.device = device

    def __call__(self, trial):
        # Suggest parameters based on algorithm
        config_params = {}
        
        if self.algo == "dqn":
            config_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "buffer_size": trial.suggest_categorical("buffer_size", [5000, 10000, 20000, 50000]),
                "learning_starts": trial.suggest_categorical("learning_starts", [100, 500, 1000]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                "exploration_fraction": trial.suggest_float("exploration_fraction", 0.1, 0.4),
                "exploration_final_eps": trial.suggest_float("exploration_final_eps", 0.01, 0.1),
                "target_update_interval": trial.suggest_categorical("target_update_interval", [100, 500, 1000, 5000])
            }
        elif self.algo == "ppo":
            n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
            # Ensure batch_size is a factor of n_steps
            possible_batches = [b for b in [16, 32, 64, 128, 256] if b <= n_steps]
            batch_size = trial.suggest_categorical("batch_size", possible_batches)
            
            config_params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
                "n_steps": n_steps,
                "batch_size": batch_size,
                "n_epochs": trial.suggest_categorical("n_epochs", [3, 5, 10, 20]),
                "gamma": trial.suggest_float("gamma", 0.90, 0.999),
                "gae_lambda": trial.suggest_float("gae_lambda", 0.80, 0.99),
                "ent_coef": trial.suggest_float("ent_coef", 1e-8, 1e-2, log=True),
                "clip_range": trial.suggest_float("clip_range", 0.1, 0.4)
            }
            
        current_cash = 100000.0
        out_of_sample_histories = []
        
        # Run Walk-Forward Validation
        for train_start, train_end, test_start, test_end in WINDOWS:
            train_df = self.df_full.loc[train_start:train_end]
            test_df = self.df_full.loc[test_start:test_end]
            
            if len(train_df) == 0 or len(test_df) == 0:
                continue
                
            train_env = TradingEnv(
                df=train_df,
                initial_cash=100000.0,
                transaction_fee=0.001,
                slippage=0.0005,
                reward_type=self.reward_type,
                feature_group=self.feature_group,
                action_space_type=self.action_space_type
            )
            
            trial_device = select_device(
                algo=self.algo,
                policy="MlpPolicy",
                batch_size=config_params.get("batch_size"),
                history_len=1,
                net_arch=config_params.get("policy_kwargs", {}).get("net_arch"),
                force_device=self.device
            )
            
            if self.algo == "dqn":
                model = DQN("MlpPolicy", train_env, verbose=0, device=trial_device, **config_params)
            elif self.algo == "ppo":
                model = PPO("MlpPolicy", train_env, verbose=0, device=trial_device, **config_params)
                
            model.learn(total_timesteps=self.timesteps)
            
            # Backtest on Test window
            test_env = TradingEnv(
                df=test_df,
                initial_cash=current_cash,
                transaction_fee=0.001,
                slippage=0.0005,
                reward_type=self.reward_type,
                feature_group=self.feature_group,
                action_space_type=self.action_space_type
            )
            
            backtester = Backtester(test_env, model)
            history_df = backtester.run_backtest()
            out_of_sample_histories.append(history_df)
            current_cash = test_env.portfolio_value
            
        if not out_of_sample_histories:
            return -10.0
            
        combined_history = pd.concat(out_of_sample_histories, ignore_index=True)
        metrics = calculate_metrics(combined_history, initial_cash=100000.0)
        
        # Penalize zero trading to encourage active search space
        if metrics.get("Number of Trades", 0) == 0:
            return -5.0
            
        sharpe = metrics.get("Sharpe Ratio", -10.0)
        if np.isnan(sharpe) or np.isinf(sharpe):
            return -10.0
            
        return float(sharpe)

def run_tuning():
    parser = argparse.ArgumentParser(description="Run walk-forward hyperparameter tuning.")
    parser.add_argument("--stock", type=str, default="reliance_ns", help="Stock ticker.")
    parser.add_argument("--algo", type=str, default="dqn", choices=["ppo", "dqn"], help="RL Algorithm.")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials.")
    parser.add_argument("--timesteps", type=int, default=30000, help="Timesteps per window.")
    parser.add_argument("--feature-group", type=str, default="state_2", help="Feature representation.")
    parser.add_argument("--reward-type", type=str, default="portfolio_return", help="Reward formulation.")
    parser.add_argument("--action-space", type=str, default="discrete_3", help="Action space configuration.")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs.")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cpu', 'cuda', 'auto'.")
    args = parser.parse_args()
    
    processed_path = f"data/processed/{args.stock}_processed.parquet"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file {processed_path} not found.")
        
    df_full = pd.read_parquet(processed_path)
    df_full.index = pd.to_datetime(df_full.index)
    df_full = df_full.sort_index()
    
    study_name = f"{args.stock}_{args.algo}_wf_study"
    
    print("\n=================== Starting Walk-Forward Tuning ===================")
    print(f"Stock: {args.stock.upper()} | Algorithm: {args.algo.upper()}")
    print(f"Trials: {args.trials} | Steps per Window: {args.timesteps}")
    print(f"Features: {args.feature_group} | Reward: {args.reward_type}")
    
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
        direction="maximize"
    )
    
    objective = WalkForwardObjective(
        stock=args.stock,
        algo=args.algo,
        df_full=df_full,
        feature_group=args.feature_group,
        reward_type=args.reward_type,
        action_space_type=args.action_space,
        timesteps=args.timesteps,
        device=args.device
    )
    
    study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)
    
    print("\n=================== Tuning Results ===================")
    try:
        print(f"Best Trial value (Combined Sharpe): {study.best_value:.4f}")
        print("Best Hyperparameters:")
        for k, v in study.best_params.items():
            print(f"  {k:30} : {v}")
            
        os.makedirs("configs", exist_ok=True)
        config_path = f"configs/{args.stock}_{args.algo}_best_params.yaml"
        
        # Load existing configs if any
        existing_configs = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                existing_configs = yaml.safe_load(f) or {}
                
        existing_configs[args.algo] = study.best_params
        
        with open(config_path, "w") as f:
            yaml.dump(existing_configs, f)
            
        print(f"Successfully saved walk-forward optimized parameters to {config_path}")
    except ValueError:
        print("No trials completed.")

if __name__ == "__main__":
    run_tuning()
