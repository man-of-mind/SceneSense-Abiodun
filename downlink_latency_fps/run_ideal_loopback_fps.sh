#!/usr/bin/env bash
# Ideal loopback FPS sweep. Assumes the host grants the requested 8 MiB UDP socket buffer
# (getsockopt after requesting 8 MiB should report ~16777216 on Linux).
set -uo pipefail

CONDITION="ideal_loopback"
TRANSPORT_LABEL="ideal_loopback_8mb"
FRONT_BIND_HOST="127.0.0.1"
BACK_BIND_HOST="127.0.0.1"
BACK_REMOTE_HOST="127.0.0.1"
BACK_RESULT_REMOTE_HOST="127.0.0.1"
START_LOCAL_BACK=1

source "$(dirname "$0")/run_common.sh"

