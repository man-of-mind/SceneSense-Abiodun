#!/usr/bin/env bash
# Launch an attach-only smoke or DG-A + DG-A.1 as a detached, self-logging stage.
# Neither mode chains DG-B.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ABIODUN_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3"
CONFIG="${SCRIPT_DIR}/configs/dg_a_v1.yaml"
OUTPUT_ROOT="${SCRIPT_DIR}/experiments"
DRY_RUN=0
PREFLIGHT_ONLY=0
ATTACH_SMOKE_REPEATS=0
ATTACH_CHANNEL_MODE="strong"
RUN_ID=""

usage() {
  echo "Usage: $0 [--dry-run|--preflight-only|--attach-smoke-repeats N] [--attach-channel-mode strong|clean] [--run-id ID] [--config PATH] [--output-root DIR]"
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
    --attach-smoke-repeats)
      ATTACH_SMOKE_REPEATS="$2"
      shift 2
      ;;
    --attach-channel-mode)
      ATTACH_CHANNEL_MODE="$2"
      shift 2
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

if [[ ! "${ATTACH_SMOKE_REPEATS}" =~ ^[0-9]+$ ]]; then
  echo "attach smoke repeat count must be a non-negative integer" >&2
  exit 2
fi
if [[ "${ATTACH_CHANNEL_MODE}" != "strong" && "${ATTACH_CHANNEL_MODE}" != "clean" ]]; then
  echo "attach channel mode must be strong or clean" >&2
  exit 2
fi
if [[ "${ATTACH_CHANNEL_MODE}" != "strong" && "${ATTACH_SMOKE_REPEATS}" -eq 0 ]]; then
  echo "clean channel mode is restricted to attach-only smoke runs" >&2
  exit 2
fi
MODE_COUNT=$((DRY_RUN + PREFLIGHT_ONLY + (ATTACH_SMOKE_REPEATS > 0 ? 1 : 0)))
if [[ "${MODE_COUNT}" -gt 1 ]]; then
  echo "choose at most one of --dry-run, --preflight-only, and --attach-smoke-repeats" >&2
  exit 2
fi

if [[ -z "${RUN_ID}" ]]; then
  if [[ "${ATTACH_SMOKE_REPEATS}" -gt 0 ]]; then
    RUN_ID="attach_smoke_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_ID="dg_a_$(date +%Y%m%d_%H%M%S)"
  fi
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
if [[ "${ATTACH_SMOKE_REPEATS}" -gt 0 ]]; then
  RUNNER_ARGS+=(
    --attach-smoke-repeats "${ATTACH_SMOKE_REPEATS}"
    --attach-channel-mode "${ATTACH_CHANNEL_MODE}"
  )
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

if [[ "${ATTACH_SMOKE_REPEATS}" -gt 0 ]]; then
  echo "Attach-only smoke detached launch accepted (${ATTACH_SMOKE_REPEATS} cold repetitions; channel=${ATTACH_CHANNEL_MODE}; D0/DG-A disabled)."
else
  echo "DG-A detached launch accepted."
fi
echo "pid=${LAUNCH_PID}"
echo "run_dir=${RUN_DIR}"
echo "launcher_log=${LAUNCH_LOG}"
echo "progress=${RUN_DIR}/progress.jsonl"
echo "completion=${RUN_DIR}/COMPLETED.json"
echo "failure=${RUN_DIR}/FAILED.json"
echo "summary=${RUN_DIR}/results_summary.json"
echo "Do not launch DG-B from this script. Re-engage only after COMPLETED.json or FAILED.json exists."
