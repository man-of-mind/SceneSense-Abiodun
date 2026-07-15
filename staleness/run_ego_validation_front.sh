#!/usr/bin/env bash
# Front-half launcher for LIVE model validation on a MOVING CAR-HEIGHT EGO (in-domain: matches training
# geometry ego_camera_z=1.55, pitch=-4, fov=120). Loopback back-half must already be running on the matching
# ports for the chosen model. Map-wide traffic (radius 200, 60 veh) so the moving ego meets NPCs <40 m.
# GT now logs actor ORIGIN (origin_x/y/z) to match the TRAINING GT convention -> validate_accuracy compares
# predictions against origin, not the bbox center. Usage: run_ego_validation_front.sh <run_group> [max_frames]
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
GROUP="${1:-val_noae_ego}"
FRAMES="${2:-600}"
echo "=== front ego validation: group=$GROUP frames=$FRAMES ==="
"$PY" staleness/carla_fusion_staleness_scenario.py \
  --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps 10 \
  --sensor-platform ego_vehicle --no-ego-freeze \
  --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
  --npc-vehicles 60 --npc-pedestrians 20 --spawn-radius 200 \
  --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
  --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "$GROUP" \
  --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
  > "staleness/front_${GROUP}.out" 2>&1
echo "  front $GROUP done rc=$?"
