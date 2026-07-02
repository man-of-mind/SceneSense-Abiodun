#!/usr/bin/env bash
# Radar-pps ablation: for each pps in {150k,200k,250k,300k}, collect the SAME 8-loop x 3-density
# moving-ego dataset (only radar pps varies; same seed+route) then train with the WINNING recipe:
#   Stage 1  seg-only Lovasz (0.5) + BN-freeze + bs24 + person_miou selection  -> vehicle IoU ~0.9
#   Stage 2  detection head (bbox2d + gated 40m + center-4 + dim-0.6) on frozen backbone
# then gated NMS-6 eval. Results appended to PPS_ABLATION_RESULTS.md.
#
# Robustness: ALL collections run in ONE CARLA session (avoids relaunch segfaults). Per-step
# failures are logged and skipped so a single hiccup doesn't sink the whole run. Phase B (training)
# is GPU-only (no CARLA).
set -uo pipefail
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
ROOT=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
VENV=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv
cd "$AB"
source "$VENV/bin/activate" 2>/dev/null || true
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:${PYTHONPATH:-}"
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
LOG=$AB/logs; mkdir -p "$LOG"
RESULTS=$AB/PPS_ABLATION_RESULTS.md
CONFIG=$AB/pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml
SEG_INIT=$AB/experiments/seg_lovasz_newdata_20260629/checkpoints/seg_lovasz_newdata_bnfreeze_bs24/best.pt
PPS_LIST=(150000 200000 250000 300000)
SEED=31
say(){ echo "[$(date '+%F %T')] $*"; }

carla_up(){ python3 -c "import carla;c=carla.Client('127.0.0.1',2000);c.set_timeout(3);c.get_world()" 2>/dev/null; }
ensure_carla(){
  carla_up && { say "CARLA already up"; return 0; }
  for attempt in 1 2 3 4; do
    say "launching CARLA (attempt $attempt)"
    ( cd "$ROOT" && setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 >/tmp/carla_ablation.log 2>&1 & )
    for i in $(seq 1 30); do carla_up && { say "CARLA up"; return 0; }; sleep 3; done
    say "CARLA launch attempt $attempt failed; killing + retrying"; pkill -9 -f CarlaUnreal 2>/dev/null; sleep 8
  done
  say "ERROR: CARLA would not start after retries"; return 1
}
teardown_carla(){ pkill -f CarlaUnreal 2>/dev/null; sleep 5; pkill -9 -f CarlaUnreal 2>/dev/null; sleep 3; say "CARLA down"; }

collect(){  # $1 pps  $2 density  $3 npc_v  $4 npc_p
  local pps=$1 den=$2 v=$3 p=$4
  local expid="moving_ego_pps${pps}_${den}_8loops_stride2"
  local ddir="$AB/fusion_training_data/$expid"
  if [[ -f "$ddir/manifest.csv" ]] && [[ $(($(wc -l < "$ddir/manifest.csv")-1)) -ge 3000 ]]; then
    say "SKIP collect $expid (already has $(($(wc -l < "$ddir/manifest.csv")-1)) rows)"; return 0; fi
  rm -rf "$ddir"
  say "COLLECT pps=$pps density=$den (v=$v p=$p)"
  python3 carla_collect_moving_ego_fusion_training_data.py \
    --sync-world --experiment-id "$expid" --seed "$SEED" --no-ego-freeze \
    --ego-autopilot-speed-difference-pct 60 --ego-follow-distance-m 28.0 --ego-ignore-lights-pct 0 \
    --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop --ego-fixed-path-min-spacing-m 3.0 \
    --ego-disable-lane-change --route-progress-every-s 2.0 --loop-return-radius-m 2.0 --loop-min-distance-m 200.0 \
    --loop-min-elapsed-s 30.0 --stop-after-loops 8 --stop-on-stuck --stuck-ignore-traffic-light-waits \
    --stuck-speed-threshold-mps 0.20 --stuck-timeout-s 60.0 --stuck-min-elapsed-s 30.0 \
    --max-samples 6000 --sample-stride 2 --warmup-ticks 30 --fps 10 \
    --camera-width 1280 --camera-height 720 --camera-fov 120 --model-input-width 768 --model-input-height 432 \
    --ego-spawn-index 80 --ego-spawn-forward-offset-m 0.0 --ego-spawn-right-offset-m 0.0 --ego-spawn-yaw-offset-deg 0.0 \
    --ego-camera-x 1.8 --ego-camera-y 0.0 --ego-camera-z 1.55 --ego-camera-pitch -4.0 --ego-camera-yaw 0.0 --ego-radar-yaw 0.0 \
    --radar-hfov 120 --radar-vfov 30 --radar-range 120 \
    --radar-points-per-second "$pps" --radar-raster-radius-px 4 --radar-temporal-window-frames 2 \
    --npc-vehicles "$v" --npc-pedestrians "$p" --npc-vehicle-speed-difference-pct 10 \
    --npc-pedestrian-max-speed-mps 0.9 --npc-pedestrian-cross-factor 0.5 --spawn-radius 80 \
    --gt-max-distance-m 140 --include-pedestrians > "$LOG/collect_pps${pps}_${den}.log" 2>&1 \
    && say "collect $expid OK ($(($(wc -l < "$ddir/manifest.csv")-1)) rows)" \
    || say "WARN collect $expid FAILED (see log)"
}

# ---------------- Phase A: all collections in one CARLA session ----------------
say "==== PPS ABLATION START. pps=${PPS_LIST[*]} seed=$SEED ===="
for pps in "${PPS_LIST[@]}"; do
  ensure_carla || { say "WARN no CARLA for pps=$pps; skipping its collection"; continue; }
  collect "$pps" low 8 10
  collect "$pps" medium 20 25
  collect "$pps" crowded 28 35
  MERGED="$AB/fusion_training_data/moving_ego_pps${pps}_merged_8loops_stride2"
  rm -rf "$MERGED"
  say "MERGE pps=$pps"
  python3 scripts/merge_fusion_training_datasets.py "$MERGED" \
    "$AB/fusion_training_data/moving_ego_pps${pps}_low_8loops_stride2" \
    "$AB/fusion_training_data/moving_ego_pps${pps}_medium_8loops_stride2" \
    "$AB/fusion_training_data/moving_ego_pps${pps}_crowded_8loops_stride2" \
    --link-mode symlink > "$LOG/merge_pps${pps}.log" 2>&1 \
    && say "merge pps=$pps OK ($(($(wc -l < "$MERGED/manifest.csv")-1)) rows)" || say "WARN merge pps=$pps FAILED"
done
teardown_carla

# ---------------- Phase B: train + eval each pps (GPU only) ----------------
[[ -f "$RESULTS" ]] || echo "# Radar-pps ablation results (seed $SEED, 8 loops x 3 densities, winning recipe)" > "$RESULTS"
for pps in "${PPS_LIST[@]}"; do
  MERGED="$AB/fusion_training_data/moving_ego_pps${pps}_merged_8loops_stride2"
  [[ -f "$MERGED/manifest.csv" ]] || { say "SKIP train pps=$pps (no merged dataset)"; continue; }
  # ---- Stage 1: seg-only Lovasz ----
  S1EXP="$AB/experiments/seg_pps${pps}"; mkdir -p "$S1EXP"
  [[ -L "$S1EXP/dataset" ]] && unlink "$S1EXP/dataset"; ln -s "$MERGED" "$S1EXP/dataset"
  say "STAGE1 seg pps=$pps"
  TRIAL=$(printf '{"name":"seg_pps%s","optimizer":"adamw","lr":0.00015,"weight_decay":0.0001,"augment_strength":"strong","geometric_augment":true,"freeze_bn":true,"input_size":[768,432],"batch_size":24,"num_workers":6,"prefetch_factor":2,"persistent_workers":false,"epochs":50,"selection_score_mode":"person_miou","class_loss_weights":[0.5,1.0,4.0],"lovasz_weight":0.5,"lr_scheduler":"cosine","lr_warmup_epochs":3,"min_lr_ratio":0.01,"poly_power":0.9,"early_stop_patience":20,"init_rgb_checkpoint":"%s","loss_weights":{"object_total":0.0}}' "$pps" "$SEG_INIT")
  python3 -m pole_lraspp_multimodal_fusion.train_fusion --config "$CONFIG" --experiment-dir "$S1EXP" \
    --trial-json "$TRIAL" --training-budget-hours 2.5 > "$LOG/seg_pps${pps}.log" 2>&1 || say "WARN stage1 pps=$pps failed"
  S1CKPT="$S1EXP/checkpoints/seg_pps${pps}/best.pt"
  [[ -f "$S1CKPT" ]] || { say "SKIP stage2 pps=$pps (no stage1 ckpt)"; continue; }
  # ---- Stage 2: detection head (winning lever-B config) ----
  say "STAGE2 det pps=$pps"
  DATASET="$MERGED" SEG_CHECKPOINT="$S1CKPT" INIT_OBJECT_CHECKPOINT="$S1CKPT" \
  RUN_NAME="det_pps${pps}" HEAD_ARCH="shared" USE_COORDCONV="false" PREDICT_BBOX2D="true" ADAPTIVE_RADIUS="true" \
  HEATMAP_RADIUS_PX="4" FREEZE_BACKBONE="true" FREEZE_CLASSIFIER="true" FREEZE_BN="true" SEG_LOSS_WEIGHT="0.0" \
  DISTILL_WEIGHT="0.0" MAX_GT_DISTANCE_M="40" EPOCHS="40" EARLY_STOP_PATIENCE="12" BATCH_SIZE="16" LR="0.0002" \
  OBJECT_LOSS_JSON='{"center":4.0,"location":1.5,"dimensions":0.6,"yaw":0.3,"parked":0.2,"radar_support":0.1,"bbox2d":1.0}' \
  bash scripts/run_arch_experiment.sh > "$LOG/det_pps${pps}.log" 2>&1 || say "WARN stage2 pps=$pps failed"
  # ---- Eval: gated NMS-6 at thr 0.10 ----
  DET="$AB/experiments/autonomous_arch_runs_20260625/det_pps${pps}/checkpoints/det_pps${pps}/best.pt"
  if [[ -f "$DET" ]]; then
    EDIR="$AB/experiments/eval_pps${pps}_nms6"; mkdir -p "$EDIR"; [[ -L "$EDIR/dataset" ]] && unlink "$EDIR/dataset"; ln -s "$MERGED" "$EDIR/dataset"
    python3 -m pole_lraspp_multimodal_fusion.evaluate_fusion --config "$CONFIG" --experiment-dir "$EDIR" \
      --checkpoint "$DET" --split test --object-score-threshold 0.10 --object-nms-radius-px 6 \
      --topk-objects 120 --match-distance-m 5.0 --max-gt-distance-m 40 --device cuda > "$LOG/eval_pps${pps}.log" 2>&1 || true
    python3 - "$pps" "$EDIR/metrics/test_fusion_evaluation_metrics.json" "$RESULTS" <<'PY' || say "WARN eval-parse pps=$pps"
import json,sys
pps,mp,res=sys.argv[1],sys.argv[2],sys.argv[3]
d=json.load(open(mp))
line=("| %s | %.3f | %.3f | %.3f | veh_iou %.3f person_iou %.3f mIoU %.3f | det F1 %.3f rec %.3f prec %.3f | xyMAE %.2f |"%(
  pps, d.get('learned_object_f1',0),d.get('learned_object_recall',0),d.get('learned_object_precision',0),
  d.get('vehicle_iou',0),d.get('person_iou',0),d.get('miou',0),
  d.get('learned_object_f1',0),d.get('learned_object_recall',0),d.get('learned_object_precision',0),
  d.get('learned_global_xy_mae_m',0)))
open(res,'a').write(line+"\n"); print(line)
PY
  fi
  say "DONE pps=$pps"
done
say "==== PPS ABLATION COMPLETE ===="
