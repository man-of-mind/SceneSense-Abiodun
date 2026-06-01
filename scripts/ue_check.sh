#!/usr/bin/env bash
# Sanity-check the UE tunnel: interface up, ping AMF + SMF over it.
set -euo pipefail
source "$(dirname "$0")/config.env"

echo "[ue_check] tunnel interface (${OAI_UE_IFACE}):"
ip -br addr show "${OAI_UE_IFACE}" || { echo "  (interface not found — UE not attached?)"; exit 1; }

echo
echo "[ue_check] ping AMF (${OAI_AMF_IP}) via ${OAI_UE_IFACE}:"
ping -I "${OAI_UE_IFACE}" -c 3 -W 2 "${OAI_AMF_IP}" || true

echo
echo "[ue_check] ping SMF (${OAI_SMF_IP}) via ${OAI_UE_IFACE}:"
ping -I "${OAI_UE_IFACE}" -c 3 -W 2 "${OAI_SMF_IP}" || true

echo
echo "[ue_check] ping ext-DN (${OAI_EXT_DN_IP}) via ${OAI_UE_IFACE}:"
ping -I "${OAI_UE_IFACE}" -c 3 -W 2 "${OAI_EXT_DN_IP}" || true
