import subprocess
import time
import os
import sys

def main():
    stocks = ["reliance_ns", "tcs_ns", "hdfcbank_ns", "infy_ns"]
    algos = ["sac", "td3", "ppo"]
    rewards = ["portfolio_return", "diff_sortino"]

    configs = []
    for stock in stocks:
        for algo in algos:
            for reward in rewards:
                configs.append((stock, algo, reward))

    # Batch size: 6 concurrent processes
    batch_size = 6

    print(f"Total configurations to run: {len(configs)}")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["PYTHONUNBUFFERED"] = "1"

    # Create logs directory if not exists
    os.makedirs("logs", exist_ok=True)

    # Run processes in batches
    for i in range(0, len(configs), batch_size):
        batch = configs[i:i+batch_size]
        print(f"\n--- Starting Batch {i//batch_size + 1} of {(len(configs)-1)//batch_size + 1} ---")
        
        processes = []
        for stock, algo, reward in batch:
            log_name = f"logs/{algo}_{stock}_{reward}_cont.log"
            print(f"Launching {algo.upper()} on {stock.upper()} with {reward}...")
            
            cmd = [
                "/opt/env/miniconda3/envs/my_exp/bin/python",
                "src/evaluation/walk_forward.py",
                "--stock", stock,
                "--feature-group", "state_7",
                "--reward-type", reward,
                "--action-space-type", "continuous",
                "--algo", algo,
                "--timesteps", "30000",
                "--device", "auto"
            ]
            
            f_log = open(log_name, "w")
            p = subprocess.Popen(cmd, stdout=f_log, stderr=subprocess.STDOUT, env=env)
            processes.append((p, stock, algo, reward, f_log))
            
        # Wait for all processes in the current batch to complete
        print("Waiting for batch to complete...")
        for p, stock, algo, reward, f_log in processes:
            p.wait()
            f_log.close()
            print(f"Finished {algo.upper()} on {stock.upper()} with {reward}.")

    print("\nAll batches completed successfully!")

if __name__ == "__main__":
    main()
