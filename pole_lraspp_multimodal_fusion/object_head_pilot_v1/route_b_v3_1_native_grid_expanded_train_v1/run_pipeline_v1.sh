#!/usr/bin/env bash
# One sequential, fail-closed chain over an already-collected corpus:
#   expanded aggregate views -> expanded v3.1 GT contract -> camera-plane localization
#   contract -> expanded symlink training view.
# No model is loaded, trained or run.  Progress is the per-phase log and the sentinel.
set -euo pipefail

ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PKG="${ROOT}/pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_native_grid_expanded_train_v1"
CAMERA_PLANE_PKG="${ROOT}/pole_lraspp_multimodal_fusion/object_head_pilot_v1/route_b_v3_1_camera_plane_contract_v1"
TS="${1:?usage: run_pipeline_v1.sh <timestamp>}"

VIEWS="${ROOT}/experiments/route_b_v3_expanded_train_views_v1/${TS}"
BASE="${ROOT}/experiments/route_b_v3_1_expanded_train_contract_v1/${TS}"
PLANE="${ROOT}/experiments/route_b_v3_1_expanded_train_camera_plane_v1/${TS}"
VIEW="${ROOT}/experiments/route_b_v3_1_native_grid_expanded_train_v1/${TS}"

REF_BASE="experiments/route_b_v3_1_clean_base_v1/20260828_012309"
REF_PLANE="experiments/route_b_v3_1_camera_plane_contract_v1/20260828_060131"

RUN="${ROOT}/experiments/route_b_v3_1_native_grid_expanded_train_v1/${TS}_pipeline"
mkdir -p "${RUN}"
LOG="${RUN}/pipeline.log"
SENTINEL="${RUN}/PIPELINE_SENTINEL"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "${LOG}"; }
fail() { log "PIPELINE_FAILED phase=$1"; echo "FAILED $1" > "${SENTINEL}"; exit 1; }

cd "${ROOT}"
unset PYTHONPATH
echo $$ > "${RUN}/pipeline.pid"
log "PIPELINE_START ts=${TS}"

log "PHASE_VIEWS_START ${VIEWS}"
/usr/bin/python3 "${PKG}/build_expanded_views_v1.py" --experiment "${VIEWS}" \
  >> "${RUN}/views.log" 2>&1 || fail views
log "PHASE_VIEWS_DONE"

log "PHASE_CONTRACT_START ${BASE}"
/usr/bin/python3 "${PKG}/build_expanded_contract_v1.py" --views "${VIEWS}" --output-root "${BASE}" \
  >> "${RUN}/contract.log" 2>&1 || fail contract
log "PHASE_CONTRACT_DONE"

log "PHASE_CAMERA_PLANE_CONFIG"
/usr/bin/python3 - "${CAMERA_PLANE_PKG}/configs/camera_plane_contract_v1.json" \
                   "${RUN}/camera_plane_contract_expanded_train_v1.json" \
                   "experiments/route_b_v3_1_expanded_train_contract_v1/${TS}" <<'PYEOF' \
  >> "${RUN}/contract.log" 2>&1 || fail camera_plane_config
import json, sys
template, out, source = sys.argv[1], sys.argv[2], sys.argv[3]
config = json.loads(open(template, encoding="utf-8").read())
config["name"] = "route_b_v3_1_expanded_train_camera_plane_v1"
config["source_experiment"] = source
# The rule and the registered v0.10 validation transition expectation are unchanged on
# purpose: validation is untouched by the six added train episodes, so the same 34/26/8/11
# expectation must still hold, and the builder's own gate proves it.
with open(out, "x", encoding="utf-8") as stream:
    json.dump(config, stream, indent=2, sort_keys=True)
    stream.write("\n")
print(json.dumps({"camera_plane_config": out, "source_experiment": source}))
PYEOF
log "PHASE_CAMERA_PLANE_START ${PLANE}"
/usr/bin/python3 "${CAMERA_PLANE_PKG}/build_contract_v1.py" --output "${PLANE}" \
  --config "${RUN}/camera_plane_contract_expanded_train_v1.json" \
  >> "${RUN}/camera_plane.log" 2>&1 || fail camera_plane
log "PHASE_CAMERA_PLANE_DONE"

log "PHASE_VIEW_START ${VIEW}"
/usr/bin/python3 "${PKG}/build_expanded_train_view_v1.py" \
  --view "${VIEW}" --contract-root "${PLANE}" --views-root "${VIEWS}" \
  --reference "camera_plane_native_grid_v3_1=${REF_PLANE}" \
  --base-comparison "clean_base=experiments/route_b_v3_1_expanded_train_contract_v1/${TS}=${REF_BASE}" \
  >> "${RUN}/view.log" 2>&1 || fail view
log "PHASE_VIEW_DONE"

VERDICT="$(cat "${VIEW}/TERMINAL_VERDICT.txt")"
log "PIPELINE_DONE verdict=${VERDICT}"
echo "DONE ${VERDICT}" > "${SENTINEL}"
