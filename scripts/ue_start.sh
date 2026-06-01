#!/usr/bin/env bash
# Start the OAI UE in rfsim mode. Run in its own terminal AFTER gNB is up.
set -euo pipefail
source "$(dirname "$0")/config.env"

cd "${OAI_RAN_BUILD}"
echo "[ue_start] starting nr-uesoftmodem against rfsim server ${UE_RFSIM_SERVER}"
sudo ./nr-uesoftmodem \
  --rfsim \
  --rfsimulator.[0].serveraddr "${UE_RFSIM_SERVER}" \
  -r "${UE_PRB}" \
  --numerology "${UE_NUMEROLOGY}" \
  --band "${UE_BAND}" \
  -C "${UE_DL_FREQ}" \
  -O "${OAI_RAN_CONF}/${UE_CONF}"
