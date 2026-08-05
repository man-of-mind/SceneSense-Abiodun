#!/usr/bin/env bash
# Track-1 uplink-only spatial-map pipeline over default OAI 106PRB.
#
# This is intentionally separate from downlink_latency_fps/run_oai_default*.sh:
# those scripts exercise the older closed-loop "return detections to the car"
# path.  This runner sends split features UE -> edge tail -> spatial-map publish
# with no result return to the car, then uses T-tracer to inspect the uplink
# traffic pattern under vanilla/default OAI link adaptation.
set -uo pipefail

AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}" || exit 2
source scripts/config.env

CLIENT="${AB}/uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py"
CHECKPOINT="${CHECKPOINT:-${AB}/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt}"
CHECKPOINT_CONTAINER="${CHECKPOINT_CONTAINER:-/work/abiodun/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt}"

CONDITION="${CONDITION:-track1_oai_default106_ttracer}"
BATCH_ID="${BATCH_ID:-$(date +%Y%m%d_%H%M%S)}"
TARGET_FPS="${TARGET_FPS:-10}"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
MAX_FRAMES="${MAX_FRAMES:-$(( FRONT_DURATION_S * TARGET_FPS ))}"
RUN_GROUP="track1_${CONDITION}_fps${TARGET_FPS}_${BATCH_ID}"
RUN_ROOT="${AB}/uplink_only_spatial_map_pipeline/runs/${CONDITION}"
RUN_DIR="${RUN_ROOT}/fps_${TARGET_FPS}_${BATCH_ID}"
CAP_ROOT="${AB}/metrics_logs/track1_oai_ttracer/${RUN_GROUP}"
LOG_ROOT="${CAP_ROOT}/logs"

TTRACER_DURATION_S="${TTRACER_DURATION_S:-420}"
GNB_CONF_DEFAULT="${GNB_CONF_DEFAULT:-${GNB_CONF}}"
UE_CONF_DEFAULT="${UE_CONF_DEFAULT:-${UE_CONF}}"
GNB_MIN_RXTXTIME="${GNB_MIN_RXTXTIME:-6}"
WAIT_TUNNEL_TRIES="${WAIT_TUNNEL_TRIES:-60}"
ENABLE_SOFTMODEM_TTRACER="${ENABLE_SOFTMODEM_TTRACER:-1}"
ATTACH_ONLY="${ATTACH_ONLY:-0}"
RECORD_GNB="${RECORD_GNB:-1}"
TTRACER_UE_PROFILE="${TTRACER_UE_PROFILE:-all}"
TTRACER_GNB_PROFILE="${TTRACER_GNB_PROFILE:-latency}"
FORCE_UL_MCS="${FORCE_UL_MCS:-}"
HOLD_MCS_FEW_SAMPLES="${HOLD_MCS_FEW_SAMPLES:-${SCENESENSE_HOLD_MCS_FEW_SAMPLES:-0}}"
MCS_POLICY="${MCS_POLICY:-${SCENESENSE_MCS_POLICY:-}}"
AIMD_MAX_DROP="${AIMD_MAX_DROP:-${SCENESENSE_AIMD_MAX_DROP:-}}"
RFSIM_CHANMOD="${RFSIM_CHANMOD:-0}"
CHANNELMOD_MODELLIST="${CHANNELMOD_MODELLIST:-}"
AWGN_PROFILE="${AWGN_PROFILE:-none}"
AWGN_NOISE_POWER_DB="${AWGN_NOISE_POWER_DB:-}"
AWGN_PLOSS_DB="${AWGN_PLOSS_DB:-}"

QUANTIZATION_MODE="${QUANTIZATION_MODE:-per_channel_uint8}"
ENTROPY_CODER="${ENTROPY_CODER:-zstd}"
ZSTD_LEVEL="${ZSTD_LEVEL:-3}"
ROI_THRESHOLD="${ROI_THRESHOLD:-0.0}"
AE_CHECKPOINT="${AE_CHECKPOINT:-}"
AE_CHECKPOINT_CONTAINER="${AE_CHECKPOINT_CONTAINER:-}"

RADAR_RASTERIZER="${RADAR_RASTERIZER:-fast}"
CAPTURE_PIPELINE="${CAPTURE_PIPELINE:-1}"
CAPTURE_PIPELINE_QUEUE_SIZE="${CAPTURE_PIPELINE_QUEUE_SIZE:-2}"
CAPTURE_PIPELINE_DROP_OLDEST="${CAPTURE_PIPELINE_DROP_OLDEST:-0}"
SENSOR_EVERY_TICK="${SENSOR_EVERY_TICK:-0}"
EDGE_DRAIN_GRACE_S="${EDGE_DRAIN_GRACE_S:-30}"
MAP_ASSUMED_PROCESS_MS="${MAP_ASSUMED_PROCESS_MS:-30}"

NPC_VEHICLES="${NPC_VEHICLES:-28}"
NPC_PEDESTRIANS="${NPC_PEDESTRIANS:-35}"
EGO_IGNORE_LIGHTS_PCT="${EGO_IGNORE_LIGHTS_PCT:-50}"
EGO_SPAWN_INDICES="${EGO_SPAWN_INDICES:-80,85,91,94,99,80}"
SPAWN_RADIUS="${SPAWN_RADIUS:-80}"
NPC_SPEED_DIFFERENCE_PCT="${NPC_SPEED_DIFFERENCE_PCT:-10}"
EGO_SPEED_DIFFERENCE_PCT="${EGO_SPEED_DIFFERENCE_PCT:-60}"
EGO_FOLLOW_DISTANCE_M="${EGO_FOLLOW_DISTANCE_M:-28.0}"

CAMERA_SOURCE_PORT="${CAMERA_SOURCE_PORT:-51001}"
REMOTE_PORT="${REMOTE_PORT:-51002}"
FRONT_SOURCE_PORT="${FRONT_SOURCE_PORT:-51003}"
BACK_SOURCE_PORT="${BACK_SOURCE_PORT:-51013}"
CAMERA_RESULT_PORT="${CAMERA_RESULT_PORT:-51004}"
SPATIAL_MAP_PORT="${SPATIAL_MAP_PORT:-39310}"
EDGE_TMP_DIR="/tmp/${RUN_GROUP}"
EDGE_TMP_CSV="${EDGE_TMP_DIR}/edge_uplink_metrics.csv"

mkdir -p "${RUN_DIR}" "${LOG_ROOT}"

SAMPLER_PID=""
UE_RECORD_PID=""
GNB_RECORD_PID=""

say() {
  echo "[$(date +%H:%M:%S)] $*" | tee -a "${CAP_ROOT}/run.log"
}

kill_pid() {
  local pid="$1"
  local name="$2"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    say "stopping ${name} pid=${pid}"
    kill -INT "${pid}" 2>/dev/null || true
    sleep 2
    kill "${pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  kill_pid "${SAMPLER_PID}" "network sampler"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
}
trap cleanup EXIT

stop_ran() {
  say "stopping existing nr-softmodem/nr-uesoftmodem processes"
  sudo pkill -x nr-uesoftmodem 2>/dev/null || true
  sudo pkill -x nr-softmodem 2>/dev/null || true
  sleep 4
}

restart_core() {
  say "restarting AMF/SMF/UPF so UE tunnel gets ${OAI_UE_IP}"
  (
    cd "${OAI_CN_DIR}" &&
      sudo docker compose restart oai-amf oai-smf oai-upf >/dev/null 2>&1
  )
  for _ in $(seq 1 45); do
    local healthy
    healthy="$(sudo docker ps --format '{{.Names}} {{.Status}}' | grep -E 'oai-(amf|smf|upf)' | grep -c healthy || true)"
    if [[ "${healthy}" -ge 3 ]]; then
      sleep 3
      return 0
    fi
    sleep 2
  done
  say "WARN: core health check did not see all AMF/SMF/UPF containers healthy"
}

start_gnb_106_default() {
  local t_args=()
  local sudo_env=()
  local chanmod_args=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_GNB_T_PORT:-2021}")
  fi
  if [[ "${RFSIM_CHANMOD}" == "1" ]]; then
    chanmod_args=(--rfsimulator.[0].options chanmod)
    if [[ -n "${CHANNELMOD_MODELLIST}" ]]; then
      chanmod_args+=(--channelmod.modellist "${CHANNELMOD_MODELLIST}")
    fi
  fi
  if [[ -n "${FORCE_UL_MCS}" || -n "${MCS_POLICY}" || -n "${AIMD_MAX_DROP}" || "${HOLD_MCS_FEW_SAMPLES}" == "1" ]]; then
    sudo_env=(env)
    if [[ -n "${FORCE_UL_MCS}" ]]; then
      sudo_env+=(SCENESENSE_FORCE_UL_MCS="${FORCE_UL_MCS}")
    fi
    if [[ -n "${MCS_POLICY}" ]]; then
      sudo_env+=(SCENESENSE_MCS_POLICY="${MCS_POLICY}")
    fi
    if [[ -n "${AIMD_MAX_DROP}" ]]; then
      sudo_env+=(SCENESENSE_AIMD_MAX_DROP="${AIMD_MAX_DROP}")
    fi
    if [[ "${HOLD_MCS_FEW_SAMPLES}" == "1" ]]; then
      sudo_env+=(SCENESENSE_HOLD_MCS_FEW_SAMPLES=1)
    fi
  fi
  say "starting default gNB: ${GNB_CONF_DEFAULT}, min_rxtxtime=${GNB_MIN_RXTXTIME}, ttracer=${ENABLE_SOFTMODEM_TTRACER}, force_ul_mcs=${FORCE_UL_MCS:-adaptive}, mcs_policy=${MCS_POLICY:-legacy}, aimd_max_drop=${AIMD_MAX_DROP:-uncapped}, hold_mcs=${HOLD_MCS_FEW_SAMPLES}, rfsim_chanmod=${RFSIM_CHANMOD}, channelmod_list=${CHANNELMOD_MODELLIST:-config-default}, awgn_profile=${AWGN_PROFILE}, awgn_noise_power_db=${AWGN_NOISE_POWER_DB:-config}, awgn_ploss_db=${AWGN_PLOSS_DB:-config}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo "${sudo_env[@]}" ./nr-softmodem \
        -O "${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}" \
        --gNBs.[0].min_rxtxtime "${GNB_MIN_RXTXTIME}" \
        --rfsim \
        "${chanmod_args[@]}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/gnb_106_default_ttracer_stdout.log" 2>&1 &
  )
}

start_ue_106() {
  local t_args=()
  local chanmod_args=()
  if [[ "${ENABLE_SOFTMODEM_TTRACER}" == "1" ]]; then
    t_args=(--T_stdout "${OAI_T_STDOUT:-2}" --T_nowait --T_port "${OAI_UE_T_PORT:-2023}")
  fi
  if [[ "${RFSIM_CHANMOD}" == "1" ]]; then
    chanmod_args=(--rfsimulator.[0].options chanmod)
    if [[ -n "${CHANNELMOD_MODELLIST}" ]]; then
      chanmod_args+=(--channelmod.modellist "${CHANNELMOD_MODELLIST}")
    fi
  fi
  say "starting single-UE softmodem: PRB=${UE_PRB}, conf=${UE_CONF_DEFAULT}, freq=${UE_DL_FREQ}, ttracer=${ENABLE_SOFTMODEM_TTRACER}, rfsim_chanmod=${RFSIM_CHANMOD}, channelmod_list=${CHANNELMOD_MODELLIST:-config-default}, awgn_profile=${AWGN_PROFILE}, awgn_noise_power_db=${AWGN_NOISE_POWER_DB:-config}, awgn_ploss_db=${AWGN_PLOSS_DB:-config}"
  (
    cd "${OAI_RAN_BUILD}" &&
      setsid nohup sudo ./nr-uesoftmodem \
        --rfsim \
        --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER:-127.0.0.1}" \
        "${chanmod_args[@]}" \
        -r "${UE_PRB}" \
        --numerology "${UE_NUMEROLOGY}" \
        --band "${UE_BAND}" \
        -C "${UE_DL_FREQ}" \
        -O "${OAI_RAN_CONF}/${UE_CONF_DEFAULT}" \
        "${t_args[@]}" \
        > "${LOG_ROOT}/ue_106_default_ttracer_stdout.log" 2>&1 &
  )
}

wait_tunnel() {
  say "waiting for ${OAI_UE_IFACE}=${OAI_UE_IP}"
  for _ in $(seq 1 "${WAIT_TUNNEL_TRIES}"); do
    if ip -4 addr show "${OAI_UE_IFACE}" 2>/dev/null | grep -q "${OAI_UE_IP}"; then
      ip -br addr show "${OAI_UE_IFACE}" | tee -a "${CAP_ROOT}/run.log"
      sleep 3
      return 0
    fi
    sleep 2
  done
  say "ERROR: ${OAI_UE_IFACE} did not attach as ${OAI_UE_IP}"
  return 1
}

start_back_half() {
  say "starting/recreating Track-1 uplink-only back-half container"
  local extra_args=(
    --uplink-only-spatial-map
    --edge-result-mode none
    --edge-metrics-csv "${EDGE_TMP_CSV}"
    --edge-receive-queue-size 32
    --spatial-map-stream
    --spatial-map-host 127.0.0.1
    --spatial-map-port "${SPATIAL_MAP_PORT}"
    --spatial-map-stream-id "${RUN_GROUP}"
    --camera-resolution custom
    --camera-width 1280
    --camera-height 720
    --camera-fov 120
    --model-input-width 768
    --model-input-height 432
    --zstd-level "${ZSTD_LEVEL}"
    --roi-threshold "${ROI_THRESHOLD}"
  )
  if [[ -n "${AE_CHECKPOINT_CONTAINER:-${AE_CHECKPOINT:-}}" ]]; then
    extra_args+=(--ae-checkpoint "${AE_CHECKPOINT_CONTAINER:-${AE_CHECKPOINT}}")
  fi
  FUSION_BACK_REMOTE_HOST="${OAI_UE_IP}" \
  FUSION_BACK_REMOTE_HOST_1="${OAI_UE_IP}" \
  FUSION_BACK_DUAL=0 \
  FUSION_BACK_SCRIPT="/work/abiodun/uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py" \
  FUSION_BACK_CHECKPOINT="${CHECKPOINT_CONTAINER}" \
  FUSION_QUANTIZATION_MODE="${QUANTIZATION_MODE}" \
  FUSION_ENTROPY_CODER="${ENTROPY_CODER}" \
  FUSION_BACK_LOG_EVERY="${FUSION_BACK_LOG_EVERY:-50}" \
  FUSION_REMOTE_PORT_1="${REMOTE_PORT}" \
  FUSION_REMOTE_SOURCE_PORT_1="${BACK_SOURCE_PORT}" \
  FUSION_CAMERA_RESULT_PORT_1="${CAMERA_RESULT_PORT}" \
  FUSION_BACK_EXTRA_ARGS="${extra_args[*]}" \
    scripts/receiver_container_fusion_back_up.sh \
    > "${LOG_ROOT}/receiver_container_track1_back_up.log" 2>&1
}

verify_back_half() {
  say "verifying Track-1 back-half container stayed up"
  sleep 8
  local running
  running="$(sudo docker inspect -f '{{.State.Running}}' oai-perception-rx 2>/dev/null || echo false)"
  if [[ "${running}" != "true" ]]; then
    say "ERROR: oai-perception-rx is not running after startup"
    sudo docker logs --tail 180 oai-perception-rx 2>&1 | tee -a "${CAP_ROOT}/run.log" || true
    return 1
  fi
  sudo docker logs --tail 120 oai-perception-rx 2>&1 | tee "${CAP_ROOT}/back_half_startup_tail.log" >/dev/null || true
}

start_network_sampler() {
  say "starting UE tunnel/network sampler"
  "${PY}" scripts/sample_oai_network_metrics.py \
    --run-group "${RUN_GROUP}" \
    --ping-host "${OAI_EXT_DN_IP}" \
    > "${LOG_ROOT}/network_sampler_stdout.log" 2>&1 &
  SAMPLER_PID=$!
}

start_ttracer_recorders() {
  say "starting UE ${TTRACER_UE_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
  scripts/ttracer_record_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source ue \
    --profile "${TTRACER_UE_PROFILE}" \
    --duration-s "${TTRACER_DURATION_S}" \
    > "${LOG_ROOT}/ttracer_record_ue_stdout.log" 2>&1 &
  UE_RECORD_PID=$!

  if [[ "${RECORD_GNB}" == "1" ]]; then
    say "starting gNB ${TTRACER_GNB_PROFILE} T-tracer recorder for ${TTRACER_DURATION_S}s"
    scripts/ttracer_record_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --duration-s "${TTRACER_DURATION_S}" \
      > "${LOG_ROOT}/ttracer_record_gnb_stdout.log" 2>&1 &
    GNB_RECORD_PID=$!
  fi
}

run_front() {
  local capture_pipeline_args=()
  if [[ "${CAPTURE_PIPELINE}" == "1" ]]; then
    capture_pipeline_args+=(--capture-pipeline --capture-pipeline-queue-size "${CAPTURE_PIPELINE_QUEUE_SIZE}")
    if [[ "${CAPTURE_PIPELINE_DROP_OLDEST}" == "1" ]]; then
      capture_pipeline_args+=(--capture-pipeline-drop-oldest)
    else
      capture_pipeline_args+=(--no-capture-pipeline-drop-oldest)
    fi
  fi
  local sensor_tick_args=(--no-sensor-every-tick)
  if [[ "${SENSOR_EVERY_TICK}" == "1" ]]; then
    sensor_tick_args=(--sensor-every-tick)
  fi
  local ae_args=()
  if [[ -n "${AE_CHECKPOINT}" ]]; then
    ae_args=(--ae-checkpoint "${AE_CHECKPOINT}")
  fi

  say "running Track-1 CARLA front fps=${TARGET_FPS} max_frames=${MAX_FRAMES} run_group=${RUN_GROUP}"
  "${PY}" "${CLIENT}" \
    --role front \
    --bind-host "${OAI_UE_IP}" \
    --remote-host "${OAI_RX_IP}" \
    --uplink-only-spatial-map \
    --edge-result-mode none \
    --sync-world \
    --fps "${TARGET_FPS}" \
    --seed 31 \
    --sensor-platform ego_vehicle \
    --no-ego-freeze \
    --ego-ignore-lights-pct "${EGO_IGNORE_LIGHTS_PCT}" \
    --ego-disable-lane-change \
    --ego-fixed-path-spawn-indices "${EGO_SPAWN_INDICES}" \
    --ego-fixed-path-loop \
    --ego-spawn-index 80 \
    --ego-spawn-z-offset-m 0.15 \
    --camera-resolution custom \
    --camera-width 1280 \
    --camera-height 720 \
    --camera-fov 120 \
    "${sensor_tick_args[@]}" \
    --model-input-width 768 \
    --model-input-height 432 \
    --ego-camera-x 1.8 \
    --ego-camera-y 0.0 \
    --ego-camera-z 1.55 \
    --ego-camera-pitch -4.0 \
    --ego-camera-yaw 0.0 \
    --ego-radar-yaw 0.0 \
    --radar-hfov 120 \
    --radar-vfov 30 \
    --radar-range 120 \
    --radar-points-per-second 200000 \
    --radar-raster-radius-px 4 \
    --radar-rasterizer "${RADAR_RASTERIZER}" \
    --radar-temporal-window-frames 2 \
    "${capture_pipeline_args[@]}" \
    --npc-vehicles "${NPC_VEHICLES}" \
    --npc-pedestrians "${NPC_PEDESTRIANS}" \
    --spawn-radius "${SPAWN_RADIUS}" \
    --npc-speed-difference-pct "${NPC_SPEED_DIFFERENCE_PCT}" \
    --fusion-checkpoint "${CHECKPOINT}" \
    --quantization-mode "${QUANTIZATION_MODE}" \
    --entropy-coder "${ENTROPY_CODER}" \
    --zstd-level "${ZSTD_LEVEL}" \
    --roi-threshold "${ROI_THRESHOLD}" \
    "${ae_args[@]}" \
    --no-spatial-map-stream \
    --headless \
    --max-frames "${MAX_FRAMES}" \
    --uplink-drain-grace-s "${EDGE_DRAIN_GRACE_S}" \
    --enable-run-logging \
    --metrics-run-dir "${RUN_DIR}/front_metrics" \
    --transport-label "${CONDITION}" \
    --run-group "${RUN_GROUP}" \
    --run-id "${RUN_GROUP}" \
    --spatial-map-stream-id "${RUN_GROUP}" \
    --camera-source-port "${CAMERA_SOURCE_PORT}" \
    --remote-port "${REMOTE_PORT}" \
    --remote-source-port "${FRONT_SOURCE_PORT}" \
    --camera-result-port "${CAMERA_RESULT_PORT}" \
    --front-device cuda \
    > "${RUN_DIR}/front_client.log" 2>&1
}

copy_edge_artifacts() {
  say "copying edge metrics/logs from oai-perception-rx"
  sudo docker cp "oai-perception-rx:${EDGE_TMP_CSV}" "${RUN_DIR}/edge_uplink_metrics.csv" >/dev/null 2>&1 || true
  sudo docker cp "oai-perception-rx:${EDGE_TMP_CSV%.csv}.summary.json" "${RUN_DIR}/edge_uplink_metrics.summary.json" >/dev/null 2>&1 || true
  sudo docker logs oai-perception-rx > "${RUN_DIR}/back_container.log" 2>&1 || true
}

postprocess() {
  say "stopping recorders before extraction"
  kill_pid "${UE_RECORD_PID}" "UE T-tracer recorder"
  UE_RECORD_PID=""
  kill_pid "${GNB_RECORD_PID}" "gNB T-tracer recorder"
  GNB_RECORD_PID=""
  kill_pid "${SAMPLER_PID}" "network sampler"
  SAMPLER_PID=""
  copy_edge_artifacts

  say "extracting UE T-tracer CSV"
  scripts/ttracer_extract_csv_smoke.sh \
    --run-group "${RUN_GROUP}" \
    --source ue \
    --profile "${TTRACER_UE_PROFILE}" \
    --clean-output \
    > "${LOG_ROOT}/ttracer_extract_ue_stdout.log" 2>&1

  if [[ "${RECORD_GNB}" == "1" ]]; then
    say "extracting gNB T-tracer CSV"
    scripts/ttracer_extract_csv_smoke.sh \
      --run-group "${RUN_GROUP}" \
      --source gnb \
      --profile "${TTRACER_GNB_PROFILE}" \
      --clean-output \
      > "${LOG_ROOT}/ttracer_extract_gnb_stdout.log" 2>&1
  fi

  say "analyzing UE grant windows"
  "${PY}" scripts/analyze_nrue_grant_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s 1.0 \
    > "${LOG_ROOT}/analyze_nrue_grant_metrics_stdout.log" 2>&1 || true

  say "analyzing UE queue/BSR windows"
  "${PY}" scripts/analyze_nrue_queue_metrics.py \
    --run-group "${RUN_GROUP}" \
    --window-s 1.0 \
    > "${LOG_ROOT}/analyze_nrue_queue_metrics_stdout.log" 2>&1 || true

  say "analyzing uplink layer latency"
  "${PY}" oai_layer_latency/analyze_uplink_layer_latency.py \
    --run-group "${RUN_GROUP}" \
    > "${LOG_ROOT}/analyze_uplink_layer_latency_stdout.log" 2>&1 || true

  cat > "${CAP_ROOT}/RUN_CONFIG.md" <<EOF
# Track-1 default-OAI t-tracer run

- run_group: \`${RUN_GROUP}\`
- gNB config: \`${GNB_CONF_DEFAULT}\`
- UE config: \`${UE_CONF_DEFAULT}\`
- UE launch: \`-r ${UE_PRB} -C ${UE_DL_FREQ}\`
- OAI behavior: force_ul_mcs=\`${FORCE_UL_MCS:-adaptive}\`, mcs_policy=\`${MCS_POLICY:-legacy}\`, aimd_max_drop=\`${AIMD_MAX_DROP:-uncapped}\`, hold_mcs=\`${HOLD_MCS_FEW_SAMPLES}\`
- RFsim channelmod: \`${RFSIM_CHANMOD}\`, profile=\`${AWGN_PROFILE}\`, noise_power_dB=\`${AWGN_NOISE_POWER_DB:-config}\`, ploss_dB=\`${AWGN_PLOSS_DB:-config}\`
- target FPS: ${TARGET_FPS}
- max frames: ${MAX_FRAMES}
- front duration budget: ${FRONT_DURATION_S}s
- radar rasterizer: ${RADAR_RASTERIZER}
- capture pipeline: ${CAPTURE_PIPELINE}, queue=${CAPTURE_PIPELINE_QUEUE_SIZE}, drop_oldest=${CAPTURE_PIPELINE_DROP_OLDEST}
- sensor_every_tick: ${SENSOR_EVERY_TICK}
- map processing assumption for reports: +${MAP_ASSUMED_PROCESS_MS} ms
- app run dir: \`${RUN_DIR}\`
- t-tracer dir: \`${AB}/metrics_logs/scenesense_ttracer/${RUN_GROUP}\`
EOF
}

say "===== START Track-1 default 106PRB OAI/T-tracer run ${RUN_GROUP} ====="
say "config: gNB=${GNB_CONF_DEFAULT}, UE_CONF=${UE_CONF_DEFAULT}, UE_PRB=${UE_PRB}, UE_FREQ=${UE_DL_FREQ}, min_rxtxtime=${GNB_MIN_RXTXTIME}, softmodem_ttracer=${ENABLE_SOFTMODEM_TTRACER}, UE_profile=${TTRACER_UE_PROFILE}, gNB_profile=${TTRACER_GNB_PROFILE}, record_gNB=${RECORD_GNB}, force_ul_mcs=${FORCE_UL_MCS:-adaptive}, mcs_policy=${MCS_POLICY:-legacy}, aimd_max_drop=${AIMD_MAX_DROP:-uncapped}, hold_mcs=${HOLD_MCS_FEW_SAMPLES}, rfsim_chanmod=${RFSIM_CHANMOD}, channelmod_list=${CHANNELMOD_MODELLIST:-config-default}, awgn_profile=${AWGN_PROFILE}, awgn_noise_power_db=${AWGN_NOISE_POWER_DB:-config}, awgn_ploss_db=${AWGN_PLOSS_DB:-config}, quant=${QUANTIZATION_MODE}, roi=${ROI_THRESHOLD}, entropy=${ENTROPY_CODER}, ae=${AE_CHECKPOINT:-none}"

if [[ ! -f "${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}" ]]; then
  say "ERROR: missing gNB config ${OAI_RAN_CONF}/${GNB_CONF_DEFAULT}"
  exit 1
fi
if [[ ! -f "${OAI_RAN_CONF}/${UE_CONF_DEFAULT}" ]]; then
  say "ERROR: missing UE config ${OAI_RAN_CONF}/${UE_CONF_DEFAULT}"
  exit 1
fi

stop_ran
restart_core
start_gnb_106_default
sleep 22
start_ue_106
if ! wait_tunnel; then
  say "gNB log: ${LOG_ROOT}/gnb_106_default_ttracer_stdout.log"
  say "UE log: ${LOG_ROOT}/ue_106_default_ttracer_stdout.log"
  exit 1
fi

if [[ "${ATTACH_ONLY}" == "1" ]]; then
  say "ATTACH_ONLY=1; attach succeeded, skipping back-half/CARLA/postprocess"
  exit 0
fi

start_back_half
verify_back_half || exit 1
start_network_sampler
start_ttracer_recorders
sleep 6

FRONT_RC=0
run_front || FRONT_RC=$?
say "front completed rc=${FRONT_RC}; waiting ${EDGE_DRAIN_GRACE_S}s for edge drain"
sleep "${EDGE_DRAIN_GRACE_S}"
postprocess

say "===== DONE Track-1 default 106PRB OAI/T-tracer run ${RUN_GROUP} rc=${FRONT_RC} ====="
say "artifacts: ${CAP_ROOT}"
exit "${FRONT_RC}"
