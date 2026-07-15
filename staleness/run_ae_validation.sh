#!/usr/bin/env bash
# Validate the 3 AE models (ae32/ae64/ae128) LIVE on loopback, moving car-height ego (in-domain), u8/ROI0,
# EXACTLY matching the no-AE validation front. Recipe (AE-attach gotcha): fusion-checkpoint = integrated
# aeN/best.pt (client drops baked-in feature_ae.*), + standalone --ae-checkpoint ae_split_aeN.pt on BOTH halves.
# For each AE: start loopback back-half -> wait for :51002 -> run front (600 fr) -> stop back-half by PID.
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
SCEN="staleness/carla_fusion_staleness_scenario.py"
FRAMES=600

for AE in 32 64 128; do
  CKPT="experiments/ae_integrated_20260710/ae${AE}/checkpoints/ae${AE}_integrated/best.pt"
  AECK="rl_agent/feature_ae/ae_split_ae${AE}.pt"
  GROUP="val_ae${AE}_ego"
  echo "================ AE-${AE} : $GROUP ================"

  # --- back-half (loopback) ---
  "$PY" "$SCEN" --role back --bind-host 127.0.0.1 --remote-host 127.0.0.1 \
    --fusion-checkpoint "$CKPT" --ae-checkpoint "$AECK" \
    --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --remote-port 51002 --remote-source-port 51013 --camera-result-port 51004 \
    --front-device cuda --back-device cuda \
    > "staleness/back_${GROUP}.out" 2>&1 &
  BACK_PID=$!
  echo "  back-half pid=$BACK_PID; waiting for :51002 ..."
  for i in $(seq 1 30); do ss -lunp 2>/dev/null | grep -q ":51002" && { echo "  back listening OK"; break; }; sleep 2; done

  # --- front (moving car-height ego, identical to no-AE run) ---
  "$PY" "$SCEN" \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps 10 \
    --sensor-platform ego_vehicle --no-ego-freeze \
    --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
    --npc-vehicles 60 --npc-pedestrians 20 --spawn-radius 200 \
    --fusion-checkpoint "$CKPT" --ae-checkpoint "$AECK" \
    --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "$GROUP" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_${GROUP}.out" 2>&1
  echo "  front $GROUP done rc=$?"

  # --- stop back-half by PID (no pkill -f: avoids self-match) ---
  kill "$BACK_PID" 2>/dev/null; sleep 2; kill -9 "$BACK_PID" 2>/dev/null
  for i in $(seq 1 10); do ss -lunp 2>/dev/null | grep -q ":51002" || break; sleep 1; done
  echo "  back-half stopped"
done
echo "AE_VALIDATION_DONE"
