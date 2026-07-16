#!/usr/bin/env bash
# Targeted regimes to fill ~22 mph (20-24 band) and push a clean ~30 mph (28-34 band). Same opportunity-window
# setup (moving car-height ego, ignore-lights, loopback no-AE). s30 uses fewer NPCs + longer run so fast cars
# reach speed on straights without pileups. Runs are auto-picked by make_speed_error_report.py (speedsweep_*).
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
CKPT="experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
SCEN="staleness/carla_fusion_staleness_scenario.py"
run_one(){
  local name="$1" spd="$2" npc="$3" frames="$4"
  echo "=== regime=$name npc_speed_diff=${spd}% npc=${npc} frames=${frames} ==="
  "$PY" "$SCEN" \
    --role front --bind-host 127.0.0.1 --remote-host 127.0.0.1 --sync-world --fps 10 \
    --sensor-platform ego_vehicle --no-ego-freeze --ego-ignore-lights-pct 100 --ego-autopilot-speed-difference-pct -30 \
    --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-x 1.8 --camera-fov 120 \
    --npc-vehicles "$npc" --npc-pedestrians 10 --spawn-radius 160 \
    --npc-speed-difference-pct "$spd" --npc-ignore-lights-pct 100 \
    --fusion-checkpoint "$CKPT" --quantization-mode per_channel_uint8 --entropy-coder zlib --roi-threshold 0.0 \
    --no-spatial-map-stream --headless --max-frames "$frames" --result-timeout 1.5 --run-group "speedsweep_${name}" \
    --camera-source-port 51001 --remote-port 51002 --remote-source-port 51003 --camera-result-port 51004 \
    > "staleness/front_speedsweep_${name}.out" 2>&1
  echo "  $name done rc=$?"
}
run_one s22 -58 70 500    # aim ~22 mph
run_one s30 -88 50 700    # aim ~30 mph (fewer NPCs, longer run)
echo "SPEED_TARGETED_DONE"
