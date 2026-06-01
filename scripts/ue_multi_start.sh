#!/usr/bin/env bash
# Start multiple OAI UEs in one nr-uesoftmodem process.
# Run in its own terminal AFTER the gNB is up.
set -euo pipefail
source "$(dirname "$0")/config.env"

UE_COUNT="${UE_COUNT:-2}"
UE_MULTI_CONF="${UE_MULTI_CONF:-ue.multi2.conf}"

cd "${OAI_RAN_BUILD}"
echo "[ue_multi_start] starting ${UE_COUNT} UE(s) using ${UE_MULTI_CONF}"
echo "[ue_multi_start] expected tunnels: oaitun_ue1=10.0.0.2, oaitun_ue2=10.0.0.3"
sudo ./nr-uesoftmodem \
  --rfsim \
  --num-ues "${UE_COUNT}" \
  -r "${UE_PRB}" \
  --numerology "${UE_NUMEROLOGY}" \
  --band "${UE_BAND}" \
  -C "${UE_DL_FREQ}" \
  -O "${OAI_RAN_CONF}/${UE_MULTI_CONF}"
