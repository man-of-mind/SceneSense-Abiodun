#!/usr/bin/env bash
# FPS half: REAL CARLA captures at 5/10/20/30 FPS on the moving car-height ego (in-domain), loopback no-AE.
# Equal sim-duration (~40s) so higher FPS = more frames = more tracker updates. Same seed + NPC regime across
# all four so the scene is comparable. Feeds make_tracked_report.py: (1) FPS-robustness of the per-frame model
# (trained @10), (2) single-frame vs Kalman-tracked error + FPS benefit. run_group = egofps_<FPS>.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
SCEN="staleness/carla_fusion_staleness_scenario.py"
DUR=40
for FPS in 5 10 20 30; do
  FRAMES=$(( DUR * FPS ))
  echo "=== egofps_${FPS}  frames=${FRAMES} (~${DUR}s sim) ==="
  "$PY" "$SCEN" \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps "$FPS" --seed 7 \
    --sensor-platform ego_vehicle --no-ego-freeze --ego-ignore-lights-pct 100 --ego-autopilot-speed-difference-pct -30 \
    --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
    --npc-vehicles 70 --npc-pedestrians 15 --spawn-radius 150 \
    --npc-speed-difference-pct -30 --npc-ignore-lights-pct 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "egofps_${FPS}" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_egofps_${FPS}.out" 2>&1
  echo "  egofps_${FPS} done rc=$?"
done
echo "EGO_FPS_CAPTURES_DONE"
