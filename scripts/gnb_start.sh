#!/usr/bin/env bash
# Start the OAI gNB in rfsim mode. Run in its own terminal — this blocks.
set -euo pipefail
source "$(dirname "$0")/config.env"

cd "${OAI_RAN_BUILD}"
echo "[gnb_start] starting nr-softmodem with ${GNB_CONF}"
sudo ./nr-softmodem \
  -O "${OAI_RAN_CONF}/${GNB_CONF}" \
  --gNBs.[0].min_rxtxtime 6 \
  --rfsim
