#!/usr/bin/env bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
export MPLCONFIGDIR=/tmp/matplotlib-cache
while pgrep -f "sweep_runner.py.*static_sweep_quant_entropy" >/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] sweep finished -> aggregating" >> rl_agent/OVERNIGHT_LOG.md
python3 rl_agent/sweep_analyze.py >> rl_agent/OVERNIGHT_LOG.md 2>&1
echo "[$(date '+%F %T')] analysis written to rl_agent/analysis/" >> rl_agent/OVERNIGHT_LOG.md
