#!/usr/bin/env bash
# Sanity-check multiple UE tunnel interfaces and core reachability.
set -euo pipefail
source "$(dirname "$0")/config.env"

UE_COUNT="${UE_COUNT:-2}"

for idx in $(seq 1 "${UE_COUNT}"); do
  iface="oaitun_ue${idx}"
  expected_ip="10.0.0.$((idx + 1))"

  echo "[ue_multi_check] tunnel interface ${iface} (expected ${expected_ip}):"
  ip -br addr show "${iface}" || { echo "  (interface not found; UE ${idx} may not be attached yet)"; exit 1; }
  echo

  echo "[ue_multi_check] ping ext-DN (${OAI_EXT_DN_IP}) via ${iface}:"
  ping -I "${iface}" -c 3 -W 2 "${OAI_EXT_DN_IP}" || true
  echo
done
