#!/usr/bin/env bash
# iperf3 UDP uplink: UE -> ext-DN.
#   ./iperf3_uplink.sh server
#   ./iperf3_uplink.sh client [bitrate] [duration]
set -euo pipefail
source "$(dirname "$0")/config.env"

mode="${1:-}"
bitrate="${2:-1M}"
duration="${3:-10}"

case "${mode}" in
  server)
    echo "[iperf3] server on ${OAI_EXT_DN_IP} (inside oai-ext-dn)"
    sudo docker exec -it oai-ext-dn iperf3 -s -B "${OAI_EXT_DN_IP}"
    ;;
  client)
    echo "[iperf3] client -> ${OAI_EXT_DN_IP} from UE IP ${OAI_UE_IP} @ ${bitrate} for ${duration}s"
    iperf3 -c "${OAI_EXT_DN_IP}" -u -b "${bitrate}" -t "${duration}" -B "${OAI_UE_IP}"
    ;;
  *)
    echo "usage: $0 {server|client} [bitrate] [duration]"
    echo "  defaults: bitrate=1M duration=10"
    exit 1
    ;;
esac
