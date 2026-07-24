#!/usr/bin/env bash
# Forced-UL-MCS CARLA experiment: start the gNB with SCENESENSE_FORCE_UL_MCS set,
# run the CARLA split-inference frontend over OAI 273PRB, record the 'latency'
# T-tracer profiles (queue/full + 4 new per-layer events), extract CSVs, analyze.
# Reuses the validated smoke recorders + run_common.sh frontend. Protects plots.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
cd "${AB}"; source scripts/config.env

FORCE_MCS="${1:-28}"
FRONT_DURATION_S="${FRONT_DURATION_S:-130}"
TTRACER_DURATION_S="${TTRACER_DURATION_S:-1200}"
GNB_CONF_273="gnb.sa.band78.fr1.273PRB.scenesense_rfsim.conf"
UE_DL_FREQ_273=3649260000; UE_SSB_273=516
BATCH_ID="forcemcs${FORCE_MCS}_$(date +%Y%m%d_%H%M%S)"
RUN_GROUP="downlink_oai_bw273_mu1_ttracer_fps10_${BATCH_ID}"
LOGDIR="/tmp/forcemcs_logs"; mkdir -p "${LOGDIR}"
echo "[forcemcs] FORCE_UL_MCS=${FORCE_MCS} run_group=${RUN_GROUP}"

UE_REC=""; GNB_REC=""
cleanup(){ [ -n "${UE_REC}" ] && kill "${UE_REC}" 2>/dev/null; [ -n "${GNB_REC}" ] && kill "${GNB_REC}" 2>/dev/null; }
trap cleanup EXIT

echo "[forcemcs] stopping RAN"
sudo pkill -x nr-uesoftmodem 2>/dev/null || true; sudo pkill -x nr-softmodem 2>/dev/null || true
until ! pgrep -x nr-softmodem >/dev/null && ! pgrep -x nr-uesoftmodem >/dev/null; do sleep 1; done

echo "[forcemcs] starting gNB with SCENESENSE_FORCE_UL_MCS=${FORCE_MCS}"
( cd "${OAI_RAN_BUILD}" && setsid nohup sudo env SCENESENSE_FORCE_UL_MCS="${FORCE_MCS}" ./nr-softmodem \
    -O "${OAI_RAN_CONF}/${GNB_CONF_273}" --gNBs.[0].min_rxtxtime 6 --rfsim \
    --T_stdout 2 --T_nowait --T_port "${OAI_GNB_T_PORT}" > "${LOGDIR}/gnb.log" 2>&1 & )
until ss -tlnp 2>/dev/null | grep -q ':4043 '; do sleep 1; done; sleep 3
echo "[forcemcs] starting UE"
( cd "${OAI_RAN_BUILD}" && setsid nohup sudo ./nr-uesoftmodem --rfsim \
    --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER}" -r 273 --numerology "${UE_NUMEROLOGY}" \
    --band "${UE_BAND}" -C "${UE_DL_FREQ_273}" --ssb "${UE_SSB_273}" \
    -O "${OAI_RAN_CONF}/ue.conf" --T_stdout 2 --T_nowait --T_port "${OAI_UE_T_PORT}" \
    > "${LOGDIR}/ue.log" 2>&1 & )
for i in $(seq 1 60); do ip -4 addr show "${OAI_UE_IFACE}" 2>/dev/null | grep -q "${OAI_UE_IP}" && break; sleep 1; done
ip -br addr show "${OAI_UE_IFACE}" || { echo "[forcemcs] UE tunnel FAILED"; exit 1; }
sleep 3

echo "[forcemcs] starting back-half container"
FUSION_BACK_REMOTE_HOST="${OAI_UE_IP}" FUSION_BACK_REMOTE_HOST_1="${OAI_UE_IP}" FUSION_BACK_DUAL=0 \
  scripts/receiver_container_downlink_fps_back_up.sh > "${LOGDIR}/backhalf.log" 2>&1

echo "[forcemcs] starting latency recorders (${TTRACER_DURATION_S}s)"
scripts/ttracer_record_smoke.sh --run-group "${RUN_GROUP}" --source ue  --profile latency --duration-s "${TTRACER_DURATION_S}" > "${LOGDIR}/rec_ue.log" 2>&1 & UE_REC=$!
scripts/ttracer_record_smoke.sh --run-group "${RUN_GROUP}" --source gnb --profile latency --duration-s "${TTRACER_DURATION_S}" > "${LOGDIR}/rec_gnb.log" 2>&1 & GNB_REC=$!
sleep 6

echo "[forcemcs] running CARLA frontend (${FRONT_DURATION_S}s sim)"
FPS_LIST=10 DURATION_S="${FRONT_DURATION_S}" CONDITION="oai_bw273_mu1_ttracer" \
TRANSPORT_LABEL="oai_bw273_forcemcs${FORCE_MCS}" FRONT_BIND_HOST="${OAI_UE_IP}" \
BACK_REMOTE_HOST="${OAI_RX_IP}" START_LOCAL_BACK=0 BATCH_ID="${BATCH_ID}" \
  bash downlink_latency_fps/run_common.sh > "${LOGDIR}/front.log" 2>&1
echo "[forcemcs] frontend done rc=$?"

kill "${UE_REC}" 2>/dev/null; wait "${UE_REC}" 2>/dev/null; UE_REC=""
kill "${GNB_REC}" 2>/dev/null; wait "${GNB_REC}" 2>/dev/null; GNB_REC=""

echo "[forcemcs] extracting"
scripts/ttracer_extract_csv_smoke.sh --run-group "${RUN_GROUP}" --source ue  --profile latency --clean-output > "${LOGDIR}/ext_ue.log" 2>&1
scripts/ttracer_extract_csv_smoke.sh --run-group "${RUN_GROUP}" --source gnb --profile latency --clean-output > "${LOGDIR}/ext_gnb.log" 2>&1

echo "[forcemcs] analyzing"
"${PY}" oai_layer_latency/analyze_uplink_layer_latency.py --run-group "${RUN_GROUP}" 2>&1 | sed -n '/## A/,/## H/p'
echo "RUN_GROUP=${RUN_GROUP}" > /tmp/forcemcs_runinfo.txt
echo "[forcemcs] DONE run_group=${RUN_GROUP}"
