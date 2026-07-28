#!/bin/bash
# Wait for the baseline (noae u8 + fp32) job to finish, then run the three knob-matrix profiles.
cd "$(dirname "$0")"
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
while pgrep -f "e5_privacy_inversion.py" >/dev/null; do sleep 20; done
for p in ae128__uint4__roi0.0 ae32__uint6__roi0.0 ae64__uint8__roi0.3; do
  echo "=== $p ==="
  $PY e5_profile_variants.py --profile "$p" --minutes 15 > "results/E5_run_$p.log" 2>&1
done
echo ALL_VARIANTS_DONE
