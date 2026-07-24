#!/usr/bin/env bash
# Validate the 4 new layer-latency T events under an iperf UDP uplink, using the
# T-tracer binaries directly (does not modify the shared smoke scripts).
#   UE  (port 2023): NR_PDCP_TX_SDU, NR_RLC_TX_SDU
#   gNB (port 2021): GNB_MAC_RX_SDU, GNB_PDCP_RX_DELIVER
set -uo pipefail
SCR="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/scripts"
source "${SCR}/config.env"
T="${OAI_T_TRACER_DIR}"; DB="${OAI_T_MESSAGES}"
OUT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/metrics_logs/scenesense_ttracer/layerval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT}"
REC=25; DUR=20; RATE="${1:-17M}"
echo "[layerval] out=${OUT}"

# iperf server
sudo docker exec -d oai-ext-dn sh -c "pkill -f 'iperf -s' 2>/dev/null; iperf -s -u -B ${OAI_EXT_DN_IP} >/tmp/iperf_srv.log 2>&1"; sleep 1

# recorders
timeout --foreground --signal=INT ${REC}s "${T}/record" -d "${DB}" -ip 127.0.0.1 -p "${OAI_UE_T_PORT}" \
  -o "${OUT}/ue.raw" -OFF -on NR_PDCP_TX_SDU -on NR_RLC_TX_SDU > "${OUT}/ue_rec.log" 2>&1 &
UEP=$!
timeout --foreground --signal=INT ${REC}s "${T}/record" -d "${DB}" -ip 127.0.0.1 -p "${OAI_GNB_T_PORT}" \
  -o "${OUT}/gnb.raw" -OFF -on GNB_MAC_RX_SDU -on GNB_PDCP_RX_DELIVER > "${OUT}/gnb_rec.log" 2>&1 &
GNP=$!

sleep 2
echo "[layerval] iperf uplink ${RATE} x ${DUR}s"
iperf -c "${OAI_EXT_DN_IP}" -u -b "${RATE}" -t "${DUR}" --bind "${OAI_UE_IP}" 2>&1 | tail -4
wait "${UEP}"; echo "[layerval] ue rec rc=$?"
wait "${GNP}"; echo "[layerval] gnb rec rc=$?"

extract() { # raw port event fields...
  local raw="$1" port="$2" ev="$3"; shift 3
  "${T}/replay" -i "${raw}" -p "${port}" > "${OUT}/${ev}_replay.log" 2>&1 & local rp=$!
  sleep 0.5
  timeout --foreground --signal=INT 20s "${T}/csv" -d "${DB}" -ip 127.0.0.1 -p "${port}" -t time "${ev}" "$@" \
    > "${OUT}/${ev}.csv" 2> "${OUT}/${ev}_csv.log"
  kill "${rp}" 2>/dev/null; wait "${rp}" 2>/dev/null
}
extract "${OUT}/ue.raw"  2203 NR_PDCP_TX_SDU      mono_sec mono_nsec ue_id rb_id sdu_bytes
extract "${OUT}/ue.raw"  2203 NR_RLC_TX_SDU       mono_sec mono_nsec ue_id rb_id sdu_bytes
extract "${OUT}/gnb.raw" 2201 GNB_MAC_RX_SDU      mono_sec mono_nsec rnti frame slot sdu_bytes
extract "${OUT}/gnb.raw" 2201 GNB_PDCP_RX_DELIVER mono_sec mono_nsec ue_id rb_id sdu_bytes

echo "=================== new-event row counts ==================="
for ev in NR_PDCP_TX_SDU NR_RLC_TX_SDU GNB_MAC_RX_SDU GNB_PDCP_RX_DELIVER; do
  n=$(( $(wc -l < "${OUT}/${ev}.csv" 2>/dev/null || echo 1) - 1 ))
  printf '%-22s %s rows\n' "${ev}" "${n}"
  head -2 "${OUT}/${ev}.csv" 2>/dev/null | sed 's/^/    /'
done
echo "OUT=${OUT}"
