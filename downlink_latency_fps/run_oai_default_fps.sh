#!/usr/bin/env bash
# Default OAI FPS sweep, front side only.
#
# Assumes:
# - OAI CN/RAN are already up.
# - UE/front host owns 10.0.0.2.
# - Back-half container is already listening at 192.168.70.140:51002 and returns to 10.0.0.2:51004.
# - OAI config is the default baseline; no TDD/QoS/PRB tuning in this study.
set -uo pipefail

CONDITION="oai_default"
TRANSPORT_LABEL="oai_default_noae"
FRONT_BIND_HOST="${FRONT_BIND_HOST:-10.0.0.2}"
BACK_REMOTE_HOST="${BACK_REMOTE_HOST:-192.168.70.140}"
START_LOCAL_BACK=0

source "$(dirname "$0")/run_common.sh"

