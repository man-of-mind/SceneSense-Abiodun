#!/usr/bin/env bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export MPLCONFIGDIR=/tmp/matplotlib-cache
while pgrep -f "sweep_runner.py.*u6" >/dev/null 2>&1; do sleep 30; done
python3 rl_agent/sweep_analyze.py >> rl_agent/analysis/accuracy_run.log 2>&1
echo "[$(date '+%F %T')] uint6 payload re-aggregated -> static_sweep_summary.md" >> rl_agent/analysis/accuracy_run.log
