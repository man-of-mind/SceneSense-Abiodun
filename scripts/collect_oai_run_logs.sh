#!/usr/bin/env bash
# Collect lightweight OAI/network snapshots into a SceneSense run directory.
#
# Usage:
#   ./collect_oai_run_logs.sh /path/to/metrics_logs/scenesense_runs/<run_id>
set -euo pipefail
source "$(dirname "$0")/config.env"

RUN_DIR="${1:-${SCENESENSE_RUN_DIR:-}}"
if [ -z "${RUN_DIR}" ]; then
    echo "usage: $0 <scenesense-run-dir>"
    echo "or set SCENESENSE_RUN_DIR"
    exit 1
fi

RUN_DIR="$(realpath -m "${RUN_DIR}")"
SNAP_DIR="${RUN_DIR}/network_snapshots"
OAI_LOG_DIR="${RUN_DIR}/oai_logs"
CORE_LOG_DIR="${OAI_LOG_DIR}/core_container_logs"
mkdir -p "${SNAP_DIR}" "${CORE_LOG_DIR}"

capture() {
    local name="$1"
    shift
    echo "[collect_oai_run_logs] ${name}: $*"
    {
        echo "# command: $*"
        echo "# captured_at: $(date --iso-8601=seconds)"
        "$@"
    } >"${SNAP_DIR}/${name}" 2>&1 || true
}

capture_text() {
    local name="$1"
    local text="$2"
    {
        echo "# captured_at: $(date --iso-8601=seconds)"
        printf "%s\n" "${text}"
    } >"${SNAP_DIR}/${name}"
}

capture_text "host.txt" "hostname=$(hostname)
date=$(date --iso-8601=seconds)
run_dir=${RUN_DIR}"

capture "ip_addr.txt" ip -br addr
capture "ip_route.txt" ip route
capture "udp_sockets.txt" ss -u -a -n
capture "oaitun_ue1_stats.txt" ip -s link show "${OAI_UE_IFACE}"
capture "oaitun_ue2_stats.txt" ip -s link show "${OAI_UE2_IFACE}"
capture "ping_ext_dn_ue1.txt" ping -I "${OAI_UE_IFACE}" -c 10 -W 1 "${OAI_EXT_DN_IP}"
capture "ping_ext_dn_ue2.txt" ping -I "${OAI_UE2_IFACE}" -c 10 -W 1 "${OAI_EXT_DN_IP}"

if command -v nvidia-smi >/dev/null 2>&1; then
    capture "nvidia_smi.txt" nvidia-smi
fi

if command -v docker >/dev/null 2>&1; then
    {
        echo "# command: sudo docker ps"
        echo "# captured_at: $(date --iso-8601=seconds)"
        sudo docker ps
    } >"${OAI_LOG_DIR}/docker_ps.txt" 2>&1 || true

    for container in \
        mysql \
        oai-amf \
        oai-smf \
        oai-upf \
        oai-ext-dn \
        oai-perception-rx; do
        echo "[collect_oai_run_logs] docker logs ${container}"
        sudo docker logs --timestamps "${container}" \
            >"${CORE_LOG_DIR}/${container}.log" 2>&1 || true
    done
fi

cat >"${OAI_LOG_DIR}/README.md" <<'EOF'
# OAI Log Notes

This folder contains lightweight network and container snapshots for the run.

For gNB/UE stdout, start those terminals with `tee` into this folder, for
example:

```bash
./gnb_start.sh 2>&1 | tee <run_dir>/oai_logs/gnb_stdout.log
./ue_multi_start.sh 2>&1 | tee <run_dir>/oai_logs/ue_stdout.log
```

For T-tracer, store raw tracer outputs under:

```text
<run_dir>/t_tracer/
```
EOF

echo "[collect_oai_run_logs] saved to ${RUN_DIR}"
