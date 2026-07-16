#!/usr/bin/env bash
# Aggressive-regime FPS captures (~30 mph target) at 5/10/20/30 FPS, to complement the existing egofps_* (mid)
# runs so each FPS covers walk->~30 mph. Same moving car-height ego, loopback no-AE, ignore-lights. run_group
# egofpsfast_<FPS>. Pooled with egofps_* by the per-speed x per-FPS analysis (which bins by MEASURED speed).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
SCEN="staleness/carla_fusion_staleness_scenario.py"
DUR=35
for FPS in 5 10 20 30; do
  FRAMES=$(( DUR * FPS ))
  echo "=== egofpsfast_${FPS}  frames=${FRAMES} (~${DUR}s sim, ~30mph regime) ==="
  "$PY" "$SCEN" \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps "$FPS" --seed 11 \
    --sensor-platform ego_vehicle --no-ego-freeze --ego-ignore-lights-pct 100 --ego-autopilot-speed-difference-pct -60 \
    --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
    --npc-vehicles 55 --npc-pedestrians 10 --spawn-radius 170 \
    --npc-speed-difference-pct -88 --npc-ignore-lights-pct 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "egofpsfast_${FPS}" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_egofpsfast_${FPS}.out" 2>&1
  echo "  egofpsfast_${FPS} done rc=$?"
done
echo "EGO_FPS_FAST_DONE"
