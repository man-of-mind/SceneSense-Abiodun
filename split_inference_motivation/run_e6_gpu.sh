#!/bin/bash
# E6 GPU arm — sweep locked GPU clocks to emulate weaker on-vehicle GPUs.
#
# Run as YOUR normal user (NOT `sudo ./run_e6_gpu.sh`) — the script calls sudo only for the
# nvidia-smi clock commands, so the Python and the output files stay owned by you.
#
#   cd .../abiodun/split_inference_motivation && ./run_e6_gpu.sh
#
# Clocks are ALWAYS reset on exit (including Ctrl-C or an error) via the trap below.
set -u
cd "$(dirname "$0")"
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
SECONDS_PER_POINT=${SECONDS_PER_POINT:-15}

# RTX 5090 supports 180–3090 MHz. These span ~17x to bracket embedded-class GPUs.
# nvidia-smi clamps to the nearest supported clock; the actual value is recorded per row.
CLOCKS=(3090 2100 1400 900 600 405 210)

cleanup() {
  echo ""
  echo "resetting GPU clocks to default..."
  sudo nvidia-smi -rgc >/dev/null 2>&1
  sudo nvidia-smi -pm 0 >/dev/null 2>&1
  nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv
}
trap cleanup EXIT INT TERM

if pgrep -f "e5_profile_variants.py|e5_privacy_inversion.py" >/dev/null; then
  echo "REFUSING TO START: an E5 training is still using the GPU."
  echo "Those runs are time-boxed at 15 min, so changing the clock mid-run would give that"
  echo "profile fewer training steps than the others and invalidate the E5 comparison."
  echo "Wait for it to finish, then re-run this script."
  trap - EXIT; exit 1
fi

echo "requesting sudo (needed only for nvidia-smi clock control)..."
sudo -v || { echo "sudo unavailable — aborting"; trap - EXIT; exit 1; }
# keep the sudo timestamp alive for the duration of the sweep
( while true; do sleep 60; sudo -n true 2>/dev/null || exit; done ) &
SUDO_KEEPALIVE=$!
trap 'kill $SUDO_KEEPALIVE 2>/dev/null; cleanup' EXIT INT TERM

sudo nvidia-smi -pm 1 >/dev/null 2>&1   # persistence mode: keeps the lock applied
rm -f results/E6_gpu_raw.csv

echo ""
echo "=== E6 GPU clock sweep (${SECONDS_PER_POINT}s sustained per config per point) ==="
for mhz in "${CLOCKS[@]}"; do
  if ! sudo nvidia-smi -lgc "${mhz},${mhz}" >/dev/null 2>&1; then
    echo "  ${mhz} MHz: could not lock — skipping"
    continue
  fi
  sleep 3   # let the clock settle before measuring
  "$PY" e6_gpu_clock_point.py --locked-mhz "$mhz" --seconds "$SECONDS_PER_POINT"
done

echo ""
echo "wrote results/E6_gpu_raw.csv"
