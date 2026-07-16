#!/usr/bin/env bash
# Opportunity-window speed sweep: moving car-height ego (in-domain) among NPC traffic whose SPEED REGIME is
# varied per run (TM percentage speed difference) + ignore-lights so cars sustain speed. Any car that enters
# good range (<~25m, in-frustum) for a few seconds is an observation; we bin by its MEASURED instantaneous
# speed afterwards. Sweeping the regime populates walking->~30mph. Loopback no-AE (clean, full delivery).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
SCEN="staleness/carla_fusion_staleness_scenario.py"
FRAMES=500
# regime name : NPC speed-difference-% (negative = faster than limit)
run_one(){
  local name="$1" spd="$2"
  echo "=== regime=$name npc_speed_diff=${spd}% ==="
  "$PY" "$SCEN" \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps 10 \
    --sensor-platform ego_vehicle --no-ego-freeze --ego-ignore-lights-pct 100 --ego-autopilot-speed-difference-pct -20 \
    --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
    --npc-vehicles 70 --npc-pedestrians 20 --spawn-radius 150 \
    --npc-speed-difference-pct "$spd" --npc-ignore-lights-pct 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$FRAMES" --result-timeout 1.5 --run-group "speedsweep_${name}" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_speedsweep_${name}.out" 2>&1
  echo "  $name done rc=$?"
}
run_one slow    60     # ~5-9 mph
run_one normal   0     # ~13-18 mph
run_one fast   -45     # ~22-28 mph
run_one veryfast -75   # ~28-34 mph (attempt)
echo "SPEED_SWEEP_DONE"
