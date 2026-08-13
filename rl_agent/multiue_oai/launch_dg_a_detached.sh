#!/usr/bin/env bash
# Launch DG-A + DG-A.1 as one detached, self-logging stage. Never chains DG-B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3"
CONFIG="${SCRIPT_DIR}/configs/dg_a_v1.yaml"
OUTPUT_ROOT="${SCRIPT_DIR}/experiments"
DRY_RUN=0
PREFLIGHT_ONLY=0
RUN_ID=""

usage() {
  echo "Usage: $0 [--dry-run|--preflight-only] [--run-id ID] [--config PATH] [--output-root DIR]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --preflight-only)
      PREFLIGHT_ONLY=1
      shift
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${DRY_RUN}" -eq 1 && "${PREFLIGHT_ONLY}" -eq 1 ]]; then
  echo "choose at most one of --dry-run and --preflight-only" >&2
  exit 2
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="dg_a_$(date +%Y%m%d_%H%M%S)"
fi
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run id must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
OUTPUT_ROOT="$(cd "${OUTPUT_ROOT}" && pwd)"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LAUNCH_LOG="${OUTPUT_ROOT}/${RUN_ID}.launcher.log"
LOCK_PATH="${OUTPUT_ROOT}/.dg_a_stage.lock"

if [[ -e "${RUN_DIR}" || -e "${LAUNCH_LOG}" ]]; then
  echo "refusing to overwrite existing run artifacts: ${RUN_DIR} / ${LAUNCH_LOG}" >&2
  exit 2
fi
if ! sudo -n true; then
  echo "sudo credentials are not cached. Run 'sudo -v', then launch again." >&2
  exit 3
fi

RUNNER_ARGS=(
  -m rl_agent.multiue_oai.runner
  --config "${CONFIG}"
  --output-dir "${RUN_DIR}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  RUNNER_ARGS+=(--dry-run)
fi
if [[ "${PREFLIGHT_ONLY}" -eq 1 ]]; then
  RUNNER_ARGS+=(--preflight-only)
fi

cd "${ABIODUN_DIR}"
setsid nohup flock -n "${LOCK_PATH}" "${PYTHON}" "${RUNNER_ARGS[@]}" \
  >"${LAUNCH_LOG}" 2>&1 < /dev/null &
LAUNCH_PID=$!
sleep 0.5
if ! kill -0 "${LAUNCH_PID}" 2>/dev/null && [[ ! -e "${RUN_DIR}/COMPLETED.json" ]]; then
  echo "detached runner exited during launch; inspect ${LAUNCH_LOG}" >&2
  exit 1
fi

echo "DG-A detached launch accepted."
echo "pid=${LAUNCH_PID}"
echo "run_dir=${RUN_DIR}"
echo "launcher_log=${LAUNCH_LOG}"
echo "progress=${RUN_DIR}/progress.jsonl"
echo "completion=${RUN_DIR}/COMPLETED.json"
echo "failure=${RUN_DIR}/FAILED.json"
echo "summary=${RUN_DIR}/results_summary.json"
echo "Do not launch DG-B from this script. Re-engage only after COMPLETED.json or FAILED.json exists."
