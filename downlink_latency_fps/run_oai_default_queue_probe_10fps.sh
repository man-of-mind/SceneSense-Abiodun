#!/usr/bin/env bash
# Default OAI fixed-offered-load queue probe, front side only.
#
# This is the backlog-pattern companion to run_oai_default_fps.sh:
# - sends one 10 FPS burst without waiting for per-frame results;
# - records async send/result event CSVs from carla_fusion_staleness_scenario.py;
# - leaves the result receiver alive after the burst so delayed results can arrive.
#
# Start the UE/gNB/CN, CARLA server, and remote back-half first.
set -uo pipefail

export FPS_LIST="${FPS_LIST:-10}"
export DURATION_S="${DURATION_S:-40}"
export QUEUE_PROBE_MODE=1
export QUEUE_PROBE_IDLE_BEFORE_S="${QUEUE_PROBE_IDLE_BEFORE_S:-10}"
export QUEUE_PROBE_COOLDOWN_S="${QUEUE_PROBE_COOLDOWN_S:-120}"
export BATCH_ID="${BATCH_ID:-oai_default_queueprobe_$(date +%Y%m%d_%H%M%S)}"

source "$(dirname "$0")/run_oai_default_fps.sh"
