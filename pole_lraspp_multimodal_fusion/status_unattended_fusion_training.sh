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

echo "run_dir=${RUN_DIR}"
if [[ -f "${RUN_DIR}/manifest.json" ]]; then
  python3 - "${RUN_DIR}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
for key in ("status", "stage", "started_at", "updated_at", "completed_at", "failed_at", "best_checkpoint", "final_report", "failure"):
    if key in payload:
        print(f"{key}={payload[key]}")
PY
else
  echo "manifest=missing"
fi

if [[ -f "${RUN_DIR}/dataset/manifest.csv" ]]; then
  echo "dataset_samples=$(($(wc -l < "${RUN_DIR}/dataset/manifest.csv") - 1))"
fi
if [[ -f "${RUN_DIR}/supervisor.pid" ]]; then
  PID="$(cat "${RUN_DIR}/supervisor.pid")"
  CMDLINE="$(ps -p "${PID}" -o args= 2>/dev/null || true)"
  if [[ "${CMDLINE}" == *"pole_lraspp_multimodal_fusion.run_pipeline"* ]]; then
    echo "supervisor_pid=${PID} running"
  elif kill -0 "${PID}" >/dev/null 2>&1; then
    echo "supervisor_pid=${PID} pid_alive_but_not_current_supervisor"
  else
    echo "supervisor_pid=${PID} not_running"
  fi
fi
if [[ -f "${RUN_DIR}/status.jsonl" ]]; then
  echo "recent_status:"
  tail -n 8 "${RUN_DIR}/status.jsonl"
fi
echo "logs:"
for path in "${RUN_DIR}/supervisor.log" "${RUN_DIR}/logs/collection.log" "${RUN_DIR}/logs/train_"*.log "${RUN_DIR}/logs/evaluate_test.log"; do
  if [[ -f "${path}" ]]; then
    echo "  ${path}"
  fi
done
