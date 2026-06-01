#!/usr/bin/env bash
# Start the OAI gNB in rfsim mode with T-tracer enabled.
# Run in its own terminal; this blocks like gnb_start.sh.
set -euo pipefail
source "$(dirname "$0")/config.env"

OAI_GNB_T_PORT="${OAI_GNB_T_PORT:-2021}"
OAI_T_STDOUT="${OAI_T_STDOUT:-2}"

cd "${OAI_RAN_BUILD}"
echo "[gnb_start_ttracer] starting nr-softmodem with ${GNB_CONF}"
echo "[gnb_start_ttracer] T-tracer port=${OAI_GNB_T_PORT}, stdout_mode=${OAI_T_STDOUT}"
sudo ./nr-softmodem \
  -O "${OAI_RAN_CONF}/${GNB_CONF}" \
  --gNBs.[0].min_rxtxtime 6 \
  --rfsim \
  --T_stdout "${OAI_T_STDOUT}" \
  --T_nowait \
  --T_port "${OAI_GNB_T_PORT}"
