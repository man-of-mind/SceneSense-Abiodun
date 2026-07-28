#!/bin/bash
cd "$(dirname "$0")"
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
# This is a CPU benchmark and does NOT need sudo. Run it as your normal user, or the
# outputs land root-owned.
if [ "$(id -u)" -eq 0 ]; then
  echo "REFUSING: run this as your normal user, not with sudo (the CPU sweep needs no root,"
  echo "and running as root leaves results/ files root-owned)."; exit 1
fi

# Refuse to run concurrently with another E6: two instances pin to the SAME first-N cores
# and measure contention instead of compute, which silently corrupts the low-thread points
# where the crossover lives.
if pgrep -f "e6_compute_crossover.py" >/dev/null; then
  echo "REFUSING: another E6 benchmark is already running (pid $(pgrep -f e6_compute_crossover.py | tr '\n' ' '))."
  echo "Running two at once invalidates both. Wait for it, or kill it and start one."; exit 1
fi

# Wait until the E5 trainings (6 dataloader workers each) are off the CPU, then let the
# machine settle, or the thread sweep measures contention not compute.
while pgrep -f "e5_profile_variants.py|e5_privacy_inversion.py" >/dev/null; do sleep 20; done
sleep 60

# Wait for the load average to settle - a loaded host makes the sweep meaningless.
# (Load decays with a ~1 min time constant, so this can take a few minutes after heavy work.)
for _ in $(seq 60); do
  LOAD1=$(cut -d' ' -f1 /proc/loadavg)
  awk "BEGIN{exit !($LOAD1 <= 2.0)}" && break
  echo "  waiting for load to settle (1-min avg $LOAD1 > 2.0)..."
  sleep 30
done
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
if awk "BEGIN{exit !($LOAD1 > 2.0)}"; then
  echo "REFUSING: load stayed at $LOAD1 after 30 min. Something else is using the CPU."; exit 1
fi
echo "load before E6: $(cat /proc/loadavg)"
$PY e6_compute_crossover.py --seconds 20 > results/E6_run.log 2>&1
echo E6_DONE
