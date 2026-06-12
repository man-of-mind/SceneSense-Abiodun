#!/usr/bin/env bash
set -euo pipefail

WORKFLOW_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/pole_lraspp_multimodal_fusion"
RUN_DIR="${1:-}"
if [[ -z "${RUN_DIR}" ]]; then
  if [[ ! -f "${WORKFLOW_ROOT}/latest_run.txt" ]]; then
    echo "No latest run pointer found at ${WORKFLOW_ROOT}/latest_run.txt" >&2
    exit 1
  fi
  RUN_DIR="$(cat "${WORKFLOW_ROOT}/latest_run.txt")"
fi
RUN_DIR="$(realpath -m "${RUN_DIR}")"
mkdir -p "${RUN_DIR}"
touch "${RUN_DIR}/stop_requested"
echo "stop_requested=${RUN_DIR}/stop_requested"

if [[ -f "${RUN_DIR}/supervisor.pid" ]]; then
  PID="$(cat "${RUN_DIR}/supervisor.pid")"
  if kill -0 "${PID}" >/dev/null 2>&1; then
    kill -TERM "${PID}"
    echo "sent_sigterm_to_supervisor=${PID}"
  else
    echo "supervisor_pid=${PID} not_running"
  fi
fi

