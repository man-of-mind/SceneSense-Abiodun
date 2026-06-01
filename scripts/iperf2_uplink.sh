#!/usr/bin/env bash
# iperf2 UDP uplink: UE -> ext-DN.
# Open two terminals; this script has two modes.
#   ./iperf2_uplink.sh server   # run on the host, starts iperf2 server in oai-ext-dn
#   ./iperf2_uplink.sh client   # run from the host (uses UE tunnel via --bind)
set -euo pipefail
source "$(dirname "$0")/config.env"

mode="${1:-}"
bitrate="${2:-10M}"

case "${mode}" in
  server)
    echo "[iperf2] server on ${OAI_EXT_DN_IP} (inside oai-ext-dn)"
    sudo docker exec -it oai-ext-dn iperf -s -i 1 -u -B "${OAI_EXT_DN_IP}"
    ;;
  client)
    echo "[iperf2] client -> ${OAI_EXT_DN_IP} from UE IP ${OAI_UE_IP} @ ${bitrate}"
    iperf -c "${OAI_EXT_DN_IP}" -u -b "${bitrate}" --bind "${OAI_UE_IP}"
    ;;
  *)
    echo "usage: $0 {server|client} [bitrate]"
    echo "  default bitrate: 10M"
    exit 1
    ;;
esac
