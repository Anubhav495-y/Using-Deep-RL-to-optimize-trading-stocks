#!/bin/bash

# Create logs directory if not exists
mkdir -p logs

echo "Launching DQN State 7 Walk-Forward campaigns..."

cd /home/vlmdg/workspace/saket

# Reliance State 7 (using its specific tuned parameters)
echo "=== RELIANCE STATE 7 ==="
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock reliance_ns \
  --feature-group state_7 \
  --reward-type portfolio_return \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/reliance_ns_dqn_best_params.yaml \
  --device auto > logs/dqn_reliance_port_state7.log 2>&1 &

/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock reliance_ns \
  --feature-group state_7 \
  --reward-type diff_sortino \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/reliance_ns_dqn_best_params.yaml \
  --device auto > logs/dqn_reliance_sortino_state7.log 2>&1 &

# TCS State 7 (using default tuned parameters)
echo "=== TCS STATE 7 ==="
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock tcs_ns \
  --feature-group state_7 \
  --reward-type portfolio_return \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_tcs_port_state7.log 2>&1 &

/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock tcs_ns \
  --feature-group state_7 \
  --reward-type diff_sortino \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_tcs_sortino_state7.log 2>&1 &

# HDFCBANK State 7 (using default tuned parameters)
echo "=== HDFCBANK STATE 7 ==="
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock hdfcbank_ns \
  --feature-group state_7 \
  --reward-type portfolio_return \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_hdfcbank_port_state7.log 2>&1 &

/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock hdfcbank_ns \
  --feature-group state_7 \
  --reward-type diff_sortino \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_hdfcbank_sortino_state7.log 2>&1 &

# INFY State 7 (using default tuned parameters)
echo "=== INFY STATE 7 ==="
/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock infy_ns \
  --feature-group state_7 \
  --reward-type portfolio_return \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_infy_port_state7.log 2>&1 &

/opt/env/miniconda3/envs/my_exp/bin/python src/evaluation/walk_forward.py \
  --stock infy_ns \
  --feature-group state_7 \
  --reward-type diff_sortino \
  --action-space-type discrete_3 \
  --algo dqn \
  --config configs/dqn_best_params.yaml \
  --device auto > logs/dqn_infy_sortino_state7.log 2>&1 &

echo "All DQN State 7 campaigns launched in the background!"
echo "Check progress by tailing logs under logs/dqn_*_state7.log"
