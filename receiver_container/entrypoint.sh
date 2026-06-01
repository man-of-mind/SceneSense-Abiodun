#!/usr/bin/env bash
# Receiver container entrypoint.
# 1. Add the return route so packets back to the UE PDN go via the UPF.
# 2. Exec whatever was passed as command (docker-compose passes the receiver script).
set -e

# Mirror oai-ext-dn's route: 10.0.0.0/16 (UE PDN range) -> UPF at 192.168.70.134.
# Without this, any reply traffic (e.g. RTCP, ICMP from the receiver) can't get
# back to the UE — the receiver-side stack would otherwise drop into the docker
# default gateway and silently fail.
ip route add 10.0.0.0/16 via 192.168.70.134 dev eth0 2>/dev/null || true
echo "[entrypoint] routes:"
ip route

echo "[entrypoint] exec: $*"
exec "$@"
