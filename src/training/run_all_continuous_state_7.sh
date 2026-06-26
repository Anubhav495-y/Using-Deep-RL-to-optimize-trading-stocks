#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

echo "Launching continuous RL validation campaigns (SAC, TD3, PPO continuous)..."

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

cd /home/vlmdg/workspace/saket

# 1. SAC (Soft Actor-Critic)
# Note: reliance_ns portfolio_return SAC is already running in background (PID 113877)

# Reliance
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_reliance_sortino_state7.log 2>&1 &

# TCS
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_tcs_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_tcs_sortino_state7.log 2>&1 &

# HDFCBANK
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_hdfcbank_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_hdfcbank_sortino_state7.log 2>&1 &

# INFY
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_infy_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo sac --timesteps 30000 --device auto > logs/sac_infy_sortino_state7.log 2>&1 &


# 2. TD3 (Twin Delayed DDPG)
# Reliance
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_reliance_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_reliance_sortino_state7.log 2>&1 &

# TCS
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_tcs_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_tcs_sortino_state7.log 2>&1 &

# HDFCBANK
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_hdfcbank_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_hdfcbank_sortino_state7.log 2>&1 &

# INFY
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_infy_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo td3 --timesteps 30000 --device auto > logs/td3_infy_sortino_state7.log 2>&1 &


# 3. PPO Continuous
# Reliance
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_reliance_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock reliance_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_reliance_sortino_state7.log 2>&1 &

# TCS
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_tcs_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock tcs_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_tcs_sortino_state7.log 2>&1 &

# HDFCBANK
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_hdfcbank_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock hdfcbank_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_hdfcbank_sortino_state7.log 2>&1 &

# INFY
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type portfolio_return --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_infy_port_state7.log 2>&1 &
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py --stock infy_ns --feature-group state_7 --reward-type diff_sortino --action-space-type continuous --algo ppo --timesteps 30000 --device auto > logs/ppo_cont_infy_sortino_state7.log 2>&1 &

echo "All continuous RL validation campaigns launched in the background!"
echo "Check progress by tailing logs under logs/sac_*, logs/td3_*, or logs/ppo_cont_*."
