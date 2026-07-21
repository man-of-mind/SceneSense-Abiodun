#!/usr/bin/env bash
# OVERNIGHT: full clean zstd knob matrix (nothing flagged/assumed).
#  Stage 1  zstd OFFLINE per-model eval (40 profiles, integrated-AE ckpts) -> sweeps_permodel_zstd   [GPU, no CARLA]
#  Stage 2  zstd FULL latency sweep (36 AE x quant x ROI) -> loopback_latency_zstd.json (full)         [CARLA]
#  Stage 3  build PERMODEL_KNOB_MATRIX_ZSTD.md (all measured)
#  Stage 4  banner + BYMODEL grouped view + CODEC_LATENCY_AB
set -uo pipefail
AB="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun"; cd "$AB"
PY="/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python"
CARLA_ROOT="/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping"
CFG="$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml"
DS="$AB/fusion_training_data/moving_ego_pps200000_merged_8loops_stride2"
S="$AB/experiments/ae_integrated_20260710/sweeps_permodel_zstd"
LBF="$AB/experiments/mprime_dropaware_20260708/sweeps_loopback_ideal_zstd_full"
LOG="$AB/rl_agent/ZSTD_FULL_OVERNIGHT_LOG.md"
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
mkdir -p "$S"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
carla_up(){ "$PY" -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" >/dev/null 2>&1; }
start_carla(){
  for a in 1 2 3; do
    log "CARLA launch attempt $a"
    ( cd "$CARLA_ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_zstd_full.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { sleep 12; carla_up && { log "CARLA up"; return 0; }; }; sleep 4; done
    pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done; return 1
}
link_ds(){ local d="$1"; mkdir -p "$d"; [[ -L "$d/dataset" ]] && unlink "$d/dataset"; ln -s "$DS" "$d/dataset"; }
declare -A CKPT=(
  [noae]="$AB/experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt"
  [ae32]="$AB/experiments/ae_integrated_20260710/ae32/checkpoints/ae32_integrated/best.pt"
  [ae64]="$AB/experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt"
  [ae128]="$AB/experiments/ae_integrated_20260710/ae128/checkpoints/ae128_integrated/best.pt")
ev(){ local name="$1"; shift; local d="$S/$name"
  [[ -f "$d/metrics/test_fusion_evaluation_metrics.json" ]] && { log "skip $name (done)"; return 0; }
  link_ds "$d"; log "eval $name"
  "$PY" -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CFG" --experiment-dir "$d" --checkpoint "$1" \
    --split test --object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 --match-distance-m 5.0 \
    --max-gt-distance-m 40 --device cuda --entropy-coder zstd "${@:2}" >> "$d/eval.log" 2>&1 || log "  WARN $name rc=$?"
}

log "===================== ZSTD FULL OVERNIGHT START ====================="
# ---- Stage 1: zstd offline eval (40 profiles) ----
log "--- Stage 1: zstd offline per-model eval ---"
for M in noae ae32 ae64 ae128; do
  CK="${CKPT[$M]}"; [[ -f "$CK" ]] || { log "MISSING ckpt $M"; continue; }
  ev "${M}__clean" "$CK"
  for Q in per_channel_uint8 per_channel_uint6 per_channel_uint4; do
    for R in 0.0 0.3 0.5; do
      ev "${M}__${Q#per_channel_}__roi${R}" "$CK" --quantization-mode "$Q" --roi-threshold "$R"
    done
  done
done
NE=$(find "$S" -name test_fusion_evaluation_metrics.json | wc -l)
log "Stage 1 done: $NE/40 eval metrics present"

# ---- Stage 2: zstd full latency sweep (36 profiles, CARLA) ----
log "--- Stage 2: zstd full latency sweep ---"
RMEM=$(sysctl -n net.core.rmem_max 2>/dev/null || echo 0)
if [ "$RMEM" -lt 8000000 ]; then log "  FATAL rmem_max=$RMEM < 8MB"; else
  CWU=0; carla_up && { CWU=1; log "reusing existing CARLA"; }
  if [ "$CWU" = "1" ] || start_carla; then
    "$PY" rl_agent/sweep_runner.py rl_agent/configs/loopback_ideal_zstd_full.json >> "$LOG" 2>&1 || log "  WARN sweep rc=$?"
    [ "$CWU" = "0" ] && { pkill -9 -f CarlaUnreal 2>/dev/null; sleep 5; }
    "$PY" rl_agent/agg_loopback.py "$LBF" rl_agent/LOOPBACK_LATENCY_ZSTD.md rl_agent/loopback_latency_zstd.json >> "$LOG" 2>&1 || log "  WARN agg"
    NL=$("$PY" -c "import json;print(len(json.load(open('rl_agent/loopback_latency_zstd.json'))))" 2>/dev/null || echo 0)
    log "Stage 2 done: $NL zstd latency profiles"
  else log "  FATAL CARLA did not start"; fi
fi

# ---- Stage 3: build the zstd matrix (all measured) ----
log "--- Stage 3: build PERMODEL_KNOB_MATRIX_ZSTD.md ---"
"$PY" rl_agent/build_knob_matrix.py "$S" rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md rl_agent/loopback_latency_zstd.json 2835 2216 >> "$LOG" 2>&1 || log "  WARN matrix"

# ---- Stage 4: banner + grouped view + A/B ----
log "--- Stage 4: banner + BYMODEL + A/B ---"
"$PY" rl_agent/apply_matrix_banner.py rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md zstd >> "$LOG" 2>&1 || log "  WARN banner"
"$PY" rl_agent/make_bymodel_grouped.py rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md rl_agent/PERMODEL_KNOB_MATRIX_ZSTD_BYMODEL.md >> "$LOG" 2>&1 || log "  WARN bymodel"
"$PY" rl_agent/make_codec_ab.py >> "$LOG" 2>&1 || log "  WARN ab"
NI=$(grep -c '~' rl_agent/PERMODEL_KNOB_MATRIX_ZSTD.md || echo -1)
log "interpolated markers in zstd matrix: $NI (expect small: header note + fp16 anchor only)"
log "===================== ZSTD FULL OVERNIGHT END ====================="
echo "ZSTD_FULL_OVERNIGHT_DONE" >> "$LOG"
