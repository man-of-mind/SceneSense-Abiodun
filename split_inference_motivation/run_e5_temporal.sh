#!/bin/bash
cd "$(dirname "$0")"
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
while pgrep -f "e5_profile_variants.py" >/dev/null; do sleep 20; done
for p in noae__uint8__roi0.0 ae128__uint4__roi0.0; do
  echo "=== $p (temporal) ==="
  $PY e5_profile_variants.py --profile "$p" --minutes 15 --split-mode temporal \
      > "results/E5_run_${p}_temporal.log" 2>&1
done
echo TEMPORAL_DONE
