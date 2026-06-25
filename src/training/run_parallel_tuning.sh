#!/bin/bash
# Create logs directory if not exists
mkdir -p logs

# 1. Reset and initialize database schema
cd /home/vlmdg/workspace/saket
rm -f optuna_wf.db
/opt/env/miniconda3/envs/my_exp/bin/python -c "import optuna; optuna.create_study(study_name='reliance_ns_dqn_wf_study', storage='sqlite:///optuna_wf.db', load_if_exists=True, direction='maximize')"

# 2. Launch 4 worker processes with restricted threads to avoid CPU thrashing
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLS_NUM_THREADS=2
export VECLIB_MAXIMUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1

echo "Launching 4 parallel workers with staggered start..."
for i in {1..4}; do
  /opt/env/miniconda3/envs/my_exp/bin/python src/training/tune_walk_forward.py \
    --stock reliance_ns \
    --algo dqn \
    --trials 8 \
    --timesteps 30000 \
    --feature-group state_2 \
    --reward-type portfolio_return \
    --device auto \
    --n-jobs 1 \
    --storage sqlite:///optuna_wf.db > logs/tune_worker_$i.log 2>&1 &
  sleep 2
done
echo "Workers launched successfully."
