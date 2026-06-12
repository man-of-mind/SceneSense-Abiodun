#!/usr/bin/env bash
set -euo pipefail

NEU_COLLAB_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab"
WORKFLOW_ROOT="${NEU_COLLAB_ROOT}/pole_lraspp_training"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3"
CONFIG="${1:-${WORKFLOW_ROOT}/configs/default_config.json}"
EXPERIMENT_DIR="${2:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="pole_lraspp_training_${STAMP}"
BACKGROUND_LOG_DIR="${WORKFLOW_ROOT}/background_logs"
mkdir -p "${BACKGROUND_LOG_DIR}"

CONFIG="$(realpath -m "${CONFIG}")"
if [[ -n "${EXPERIMENT_DIR}" ]]; then
  EXPERIMENT_DIR="$(realpath -m "${EXPERIMENT_DIR}")"
fi

CMD="cd '${WORKFLOW_ROOT}' && PYTHONPATH='${WORKFLOW_ROOT}:${NEU_COLLAB_ROOT}:${PYTHONPATH:-}' MPLBACKEND=Agg PYTHONUNBUFFERED=1 '${PYTHON}' -m pole_lraspp_training.run_pipeline --config '${CONFIG}'"
if [[ -n "${EXPERIMENT_DIR}" ]]; then
  CMD="${CMD} --experiment-dir '${EXPERIMENT_DIR}'"
fi

if command -v screen >/dev/null 2>&1; then
  screen -L -Logfile "${BACKGROUND_LOG_DIR}/${SESSION}.screen.log" -dmS "${SESSION}" bash -lc "${CMD}"
  echo "screen_session=${SESSION}"
  echo "attach_command=screen -r ${SESSION}"
  echo "screen_log=${BACKGROUND_LOG_DIR}/${SESSION}.screen.log"
  if [[ -n "${EXPERIMENT_DIR}" ]]; then
    echo "experiment_dir=${EXPERIMENT_DIR}"
  fi
elif command -v nohup >/dev/null 2>&1; then
  nohup bash -lc "${CMD}" >"${BACKGROUND_LOG_DIR}/${SESSION}.log" 2>&1 &
  echo "nohup_pid=$!"
  echo "nohup_log=${BACKGROUND_LOG_DIR}/${SESSION}.log"
  if [[ -n "${EXPERIMENT_DIR}" ]]; then
    echo "experiment_dir=${EXPERIMENT_DIR}"
  fi
else
  echo "Neither screen nor nohup is available." >&2
  exit 1
fi
