#!/usr/bin/env bash
set -euo pipefail

NEU_COLLAB_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab"
WORKFLOW_ROOT="${NEU_COLLAB_ROOT}/pole_lraspp_multimodal_fusion"
RGB_WORKFLOW_ROOT="${NEU_COLLAB_ROOT}/pole_lraspp_training"
PYTHON="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3"
CARLA_BIN="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/CarlaUnreal.sh"
CONFIG="${WORKFLOW_ROOT}/configs/fusion_full_run.yaml"
MODE="direct"
RESUME="auto"
EXPERIMENT_DIR=""
RUNTIME_BUDGET_HOURS=""
SESSION_NAME="pole_lraspp_fusion"
DRY_RUN=0

usage() {
  cat <<'USAGE'
Usage: launch_unattended_fusion_training.sh [options]

Options:
  --config PATH              Fusion config YAML/JSON.
  --python PATH              Python interpreter.
  --carla-bin PATH           CarlaUnreal.sh path.
  --experiment-dir PATH      Existing or new experiment directory.
  --runtime-budget-hours N   Override runtime budget.
  --resume auto|off          Resume from manifest state when possible.
  --mode direct|screen|nohup Launch mode. Default: direct.
  --session-name NAME        screen session name for --mode screen.
  --dry-run                  Create run directory and validate args only.
  -h, --help                 Show this help.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --carla-bin) CARLA_BIN="$2"; shift 2 ;;
    --experiment-dir) EXPERIMENT_DIR="$2"; shift 2 ;;
    --runtime-budget-hours) RUNTIME_BUDGET_HOURS="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --session-name) SESSION_NAME="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${MODE}" != "direct" && "${MODE}" != "screen" && "${MODE}" != "nohup" ]]; then
  echo "--mode must be direct, screen, or nohup" >&2
  exit 2
fi
if [[ "${RESUME}" != "auto" && "${RESUME}" != "off" ]]; then
  echo "--resume must be auto or off" >&2
  exit 2
fi

CONFIG="$(realpath -m "${CONFIG}")"
# Do not resolve the venv Python symlink to /usr/bin/python3.x. Executing the
# symlink target directly bypasses the virtualenv's site-packages, which breaks
# imports such as cv2 even when they are installed in the venv.
if [[ "${PYTHON}" != /* ]]; then
  PYTHON="$(realpath -m "${PYTHON}")"
fi
CARLA_BIN="$(realpath -m "${CARLA_BIN}")"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Python is not executable: ${PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 1
fi
if [[ ! -x "${CARLA_BIN}" ]]; then
  echo "CARLA binary is not executable: ${CARLA_BIN}" >&2
  exit 1
fi

if [[ -z "${EXPERIMENT_DIR}" ]]; then
  STAMP="$(date +%Y%m%d_%H%M%S)"
  EXPERIMENT_DIR="${NEU_COLLAB_ROOT}/experiments/pole_lraspp_multimodal_fusion/${STAMP}_pole_lraspp_multimodal_fusion_learned_localization"
fi
EXPERIMENT_DIR="$(realpath -m "${EXPERIMENT_DIR}")"
mkdir -p "${EXPERIMENT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%s\n' "${EXPERIMENT_DIR}" > "${WORKFLOW_ROOT}/latest_dry_run.txt"
else
  printf '%s\n' "${EXPERIMENT_DIR}" > "${WORKFLOW_ROOT}/latest_run.txt"
fi

SCRIPT_PATH="$(realpath -m "$0")"
BACKGROUND_LOG_DIR="${WORKFLOW_ROOT}/background_logs"
mkdir -p "${BACKGROUND_LOG_DIR}"
DIRECT_ARGS=(
  --mode direct
  --config "${CONFIG}"
  --python "${PYTHON}"
  --carla-bin "${CARLA_BIN}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --resume "${RESUME}"
)
if [[ -n "${RUNTIME_BUDGET_HOURS}" ]]; then
  DIRECT_ARGS+=(--runtime-budget-hours "${RUNTIME_BUDGET_HOURS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  DIRECT_ARGS+=(--dry-run)
fi

if [[ "${MODE}" == "screen" ]]; then
  if ! command -v screen >/dev/null 2>&1; then
    echo "screen is not available on this host." >&2
    exit 1
  fi
  SCREEN_LOG="${BACKGROUND_LOG_DIR}/${SESSION_NAME}.screen.log"
  screen -L -Logfile "${SCREEN_LOG}" -dmS "${SESSION_NAME}" bash "${SCRIPT_PATH}" "${DIRECT_ARGS[@]}"
  echo "screen_session=${SESSION_NAME}"
  echo "attach_command=screen -r ${SESSION_NAME}"
  echo "screen_log=${SCREEN_LOG}"
  echo "experiment_dir=${EXPERIMENT_DIR}"
  exit 0
fi

if [[ "${MODE}" == "nohup" ]]; then
  NOHUP_LOG="${BACKGROUND_LOG_DIR}/${SESSION_NAME}.nohup.log"
  nohup bash "${SCRIPT_PATH}" "${DIRECT_ARGS[@]}" >"${NOHUP_LOG}" 2>&1 &
  echo "nohup_pid=$!"
  echo "nohup_log=${NOHUP_LOG}"
  echo "experiment_dir=${EXPERIMENT_DIR}"
  exit 0
fi

cd "${WORKFLOW_ROOT}"
export PYTHONPATH="${WORKFLOW_ROOT}:${RGB_WORKFLOW_ROOT}:${NEU_COLLAB_ROOT}:${PYTHONPATH:-}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONUNBUFFERED=1

CMD=(
  "${PYTHON}" -m pole_lraspp_multimodal_fusion.run_pipeline
  --config "${CONFIG}"
  --experiment-dir "${EXPERIMENT_DIR}"
  --carla-bin "${CARLA_BIN}"
  --resume "${RESUME}"
)
if [[ -n "${RUNTIME_BUDGET_HOURS}" ]]; then
  CMD+=(--runtime-budget-hours "${RUNTIME_BUDGET_HOURS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

echo "experiment_dir=${EXPERIMENT_DIR}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "latest_dry_run=${WORKFLOW_ROOT}/latest_dry_run.txt"
else
  echo "latest_run=${WORKFLOW_ROOT}/latest_run.txt"
fi
echo "command=${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "${EXPERIMENT_DIR}/launcher.log"
exit "${PIPESTATUS[0]}"
