import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import pandas as pd
import numpy as np
from src.environment.trading_env import TradingEnv

def benchmark():
    processed_file = "data/processed/reliance_ns_processed.parquet"
    if not os.path.exists(processed_file):
        print(f"Error: {processed_file} does not exist. Run process_all.py first.")
        return
        
    df = pd.read_parquet(processed_file)
    
    env = TradingEnv(
        df=df,
        initial_cash=100000.0,
        transaction_fee=0.001,
        slippage=0.0005,
        reward_type="portfolio_return"
    )
    
    # Run 100 episodes
    start = time.time()
    total_steps = 0
    
    for i in range(100):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1
            
    end = time.time()
    duration = end - start
    sps = total_steps / duration
    print(f"Executed {total_steps} steps in {duration:.4f}s ({sps:.2f} steps/second)")

if __name__ == "__main__":
    benchmark()
