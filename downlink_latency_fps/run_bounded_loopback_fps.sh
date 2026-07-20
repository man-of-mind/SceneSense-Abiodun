#!/usr/bin/env bash
# Bounded/default-buffer loopback FPS sweep.
#
# This temporarily restores the old net.core rmem/wmem cap (212992 bytes),
# runs the same loopback sweep, then restores the values observed at script start.
# Use only when we deliberately want the bounded-buffer/historical loopback condition.
set -uo pipefail

if [[ "${CONFIRM_SYSCTL:-0}" != "1" ]]; then
  echo "Refusing to change host sysctl without CONFIRM_SYSCTL=1."
  echo "Run as: CONFIRM_SYSCTL=1 bash downlink_latency_fps/run_bounded_loopback_fps.sh"
  exit 2
fi

read_sysctl() {
  sysctl -n "$1"
}

set_sysctl() {
  sudo sysctl -w "$1=$2" >/dev/null
}

ORIG_RMEM="$(read_sysctl net.core.rmem_max)"
ORIG_WMEM="$(read_sysctl net.core.wmem_max)"

restore_sysctl() {
  echo "Restoring net.core.rmem_max=$ORIG_RMEM net.core.wmem_max=$ORIG_WMEM"
  set_sysctl net.core.rmem_max "$ORIG_RMEM" || true
  set_sysctl net.core.wmem_max "$ORIG_WMEM" || true
}

after_run_common() {
  restore_sysctl
}

echo "Setting bounded loopback caps: net.core.rmem_max=212992 net.core.wmem_max=212992"
set_sysctl net.core.rmem_max 212992
set_sysctl net.core.wmem_max 212992

CONDITION="bounded_loopback"
TRANSPORT_LABEL="bounded_loopback_208kb"
FRONT_BIND_HOST="127.0.0.1"
BACK_BIND_HOST="127.0.0.1"
BACK_REMOTE_HOST="127.0.0.1"
BACK_RESULT_REMOTE_HOST="127.0.0.1"
START_LOCAL_BACK=1

source "$(dirname "$0")/run_common.sh"
