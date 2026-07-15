#!/usr/bin/env bash
# Real CARLA captures at several FPS (pole + NPCs, loopback). Equal sim-duration (~25 s) so higher FPS =
# more frames = more tracker updates. Verifies per-frame model accuracy is FPS-robust + feeds the tracker.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
DUR=25
for FPS in 5 10 20 30; do
  FRAMES=$(( DUR * FPS ))
  echo "=== FPS=$FPS frames=$FRAMES (~${DUR}s sim) ==="
  "$PY" staleness/carla_fusion_staleness_scenario.py \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps "$FPS" \
    --traffic-light-id 14 --camera-x 9 --camera-y 2 --camera-pitch -30 --camera-yaw-offset 50 --camera-roll 0 --camera-fov 100 \
    --npc-vehicles 40 --npc-pedestrians 20 --spawn-radius 55 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "fps_${FPS}" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_fps_${FPS}.out" 2>&1
  echo "  FPS=$FPS done rc=$?"
done
echo "FPS_CAPTURES_DONE"
