#!/usr/bin/env bash
# Show CN container status, and the IP that AMF/SMF reported to the UE.
set -euo pipefail
source "$(dirname "$0")/config.env"

echo "[cn_status] container state:"
sudo docker ps --filter "name=oai-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo
echo "[cn_status] AMF — IP/registration hints:"
sudo docker logs oai-amf 2>&1 | grep -Ei "IPV4|UE Address|Registered|5GMM-REGISTERED" | tail -10 || true

echo
echo "[cn_status] SMF — assigned UE IP:"
sudo docker logs oai-smf 2>&1 | grep -Ei "UE Address|10\\.0\\.0|IPv4 Address" | tail -10 || true
