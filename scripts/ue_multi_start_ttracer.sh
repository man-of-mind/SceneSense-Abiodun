#!/usr/bin/env bash
# Start multiple OAI UEs in one nr-uesoftmodem process with T-tracer enabled.
# Run in its own terminal after the T-enabled gNB is up.
set -euo pipefail
source "$(dirname "$0")/config.env"

UE_COUNT="${UE_COUNT:-2}"
UE_MULTI_CONF="${UE_MULTI_CONF:-ue.multi2.conf}"
OAI_UE_T_PORT="${OAI_UE_T_PORT:-2023}"
OAI_T_STDOUT="${OAI_T_STDOUT:-2}"

cd "${OAI_RAN_BUILD}"
echo "[ue_multi_start_ttracer] starting ${UE_COUNT} UE(s) using ${UE_MULTI_CONF}"
echo "[ue_multi_start_ttracer] expected tunnels: oaitun_ue1=10.0.0.2, oaitun_ue2=10.0.0.3"
echo "[ue_multi_start_ttracer] T-tracer port=${OAI_UE_T_PORT}, stdout_mode=${OAI_T_STDOUT}"
sudo ./nr-uesoftmodem \
  --rfsim \
  --num-ues "${UE_COUNT}" \
  -r "${UE_PRB}" \
  --numerology "${UE_NUMEROLOGY}" \
  --band "${UE_BAND}" \
  -C "${UE_DL_FREQ}" \
  -O "${OAI_RAN_CONF}/${UE_MULTI_CONF}" \
  --T_stdout "${OAI_T_STDOUT}" \
  --T_nowait \
  --T_port "${OAI_UE_T_PORT}"
