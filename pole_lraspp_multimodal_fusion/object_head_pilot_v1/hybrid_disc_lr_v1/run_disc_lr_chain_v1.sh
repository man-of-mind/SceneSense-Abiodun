#!/usr/bin/env bash
# Option A: one registered discriminative-learning-rate follow-up.
#
#   freeze audit -> six epochs -> decode epochs 3 and 6 at score 0.20 and 0.02 -> gate
#
# Create-only. Architecture, loss, targets, evaluator and decoder are reused
# unchanged; the canonical baseline is NOT re-decoded. Test split never opened.
set -u

ABIODUN=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
PKG=$ABIODUN/pole_lraspp_multimodal_fusion
HERE=$PKG/object_head_pilot_v1/hybrid_disc_lr_v1
DECODER=$PKG/object_head_pilot_v1/hybrid_centerfusion_v1/evaluate_hybrid_route_b_v1.py
EXP=$(cat "$HERE/EXP_DIR.txt")
WARM=$ABIODUN/experiments/hybrid_centerfusion_v1/20260826_162833/checkpoints/hybrid_centerfusion_v1/warm_start.pt
WARM_SHA=935d7c10afe10580afbda7b5691a9985358b846c421199fb99fd761abacd424b
CFG=$PKG/object_head_pilot_v1/configs/route_b_noae_precision_pilot_v1.yaml
TRIAL_JSON=$HERE/configs/disc_lr_v1.json
TRIAL=disc_lr_v1
PY=/usr/bin/python3
LANES=2
CKDIR=$EXP/checkpoints/$TRIAL

cd "$PKG"
LOG=$EXP/chain.log
log(){ echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }
notify(){ command -v notify-send >/dev/null 2>&1 && notify-send "disc-LR follow-up" "$1" || true; log "NOTIFY: $1"; }
finish(){ echo "$1" > "$EXP/TERMINAL_VERDICT.txt"; notify "$1"; log "CHAIN_DONE verdict=$1"; exit 0; }

mklane(){ local lane="$EXP/lane$1"
  if [ ! -e "$lane" ]; then mkdir -p "$lane"; ln -s "$EXP/dataset" "$lane/dataset"; ln -s "$EXP/provenance" "$lane/provenance"; fi
  echo "$lane"; }

decode(){ local lane=$1 ckpt=$2 tag=$3 score=$4
  [ -d "$EXP/eval/$tag" ] && { log "skip $tag"; return 0; }
  [ -f "$ckpt" ] || { log "MISSING $ckpt"; return 1; }
  local t0=$(date +%s)
  "$PY" "$DECODER" --experiment-dir "$lane" --checkpoint "$ckpt" --tag "$tag" --config "$CFG" \
      --split val --feature-drop-fraction 0.0 --object-score-threshold "$score" \
      > "$lane/${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then mkdir -p "$EXP/eval"; mv "$lane/eval/$tag" "$EXP/eval/$tag"
    log "decoded $tag in $(( $(date +%s) - t0 ))s"
  else log "DECODE_FAILED $tag rc=$rc"; tail -20 "$lane/${tag}.log" | tee -a "$LOG"; fi
  return $rc; }

run_grid(){ mapfile -t JOBS
  local i j lane
  for i in $(seq 0 $((LANES-1))); do
    lane=$(mklane "$i")
    ( for j in $(seq "$i" "$LANES" $(( ${#JOBS[@]} - 1 )) ); do
        IFS=$'\t' read -r ck tag score <<< "${JOBS[$j]}"; decode "$lane" "$ck" "$tag" "$score"
      done ) &
  done
  wait; }

# ------------------------------------------------------- freeze audit + training
log "phase 1: freeze audit and six epochs at lr 3e-4 (inherited frozen)"
"$PY" "$HERE/disc_lr_train_entry_v1.py" --config "$CFG" --trial-json "$TRIAL_JSON" \
    --experiment-dir "$EXP" --warm-start-state "$WARM" --warm-start-sha256 "$WARM_SHA" \
    --training-budget-hours 0 >> "$EXP/train_stdout.log" 2>&1
RC=$?
if [ $RC -eq 4 ] || [ $RC -eq 3 ]; then
  log "freeze audit or warm-start hash check failed (rc=$RC)"; finish IMPLEMENTATION_BLOCKED
fi
if [ $RC -ne 0 ]; then log "training returned rc=$RC"; finish IMPLEMENTATION_BLOCKED; fi
for ep in 003 006; do
  [ -f "$CKDIR/epoch_$ep.pt" ] || { log "missing epoch_$ep.pt"; finish IMPLEMENTATION_BLOCKED; }
done

# ------------------------------------------------------------------- evaluation
log "phase 2: decode epochs 3 and 6 at score 0.20 and 0.02"
printf '%s\t%s\t%s\n' \
  "$CKDIR/epoch_003.pt" "disc_lr_ep003_s020" "0.20" \
  "$CKDIR/epoch_003.pt" "disc_lr_ep003_s002" "0.02" \
  "$CKDIR/epoch_006.pt" "disc_lr_ep006_s020" "0.20" \
  "$CKDIR/epoch_006.pt" "disc_lr_ep006_s002" "0.02" | run_grid

# ------------------------------------------------------------------------- gate
log "phase 3: per-checkpoint gate against the canonical baseline"
"$PY" - "$EXP" <<'PYEOF' >> "$EXP/gate_stdout.log" 2>&1
import json, math, sys
from pathlib import Path

EXP = Path(sys.argv[1])
# Canonical baseline, already verified; deliberately NOT re-decoded.
BASE = {"s002_vehicle_recall": 0.5702, "s002_person_recall": 0.4852,
        "s020_vehicle_precision": 0.4624, "s020_person_precision": 0.3480, "miou": 0.7078}
REQ = {"recall_gain": 0.05, "precision_drop": 0.03, "miou_drop": 0.02}


def load(tag):
    d = EXP / "eval" / tag
    derived = json.loads((d / "derived_metrics.json").read_text())
    evaluator = json.loads((d / "evaluator_metrics.json").read_text())
    p = derived["primary"]
    out = {"tag": tag, "checkpoint": derived["checkpoint"],
           "checkpoint_sha256": derived.get("checkpoint_sha256", "")}
    for cls in ("vehicle", "person"):
        for m in ("precision", "recall", "f1", "xy_mae_m"):
            out[f"{cls}_{m}"] = float(p[f"{cls}_{m}"])
    for k in ("miou", "vehicle_iou", "person_iou"):
        out[k] = float(evaluator[k])
    return out


rows = []
for epoch in (3, 6):
    s020 = load(f"disc_lr_ep{epoch:03d}_s020")
    s002 = load(f"disc_lr_ep{epoch:03d}_s002")
    checks = [
        ("s002_vehicle_recall_gain", s002["vehicle_recall"] - BASE["s002_vehicle_recall"],
         REQ["recall_gain"], s002["vehicle_recall"] - BASE["s002_vehicle_recall"] >= REQ["recall_gain"]),
        ("s002_person_recall_gain", s002["person_recall"] - BASE["s002_person_recall"],
         REQ["recall_gain"], s002["person_recall"] - BASE["s002_person_recall"] >= REQ["recall_gain"]),
        ("s020_vehicle_precision_drop", BASE["s020_vehicle_precision"] - s020["vehicle_precision"],
         REQ["precision_drop"], BASE["s020_vehicle_precision"] - s020["vehicle_precision"] <= REQ["precision_drop"]),
        ("s020_person_precision_drop", BASE["s020_person_precision"] - s020["person_precision"],
         REQ["precision_drop"], BASE["s020_person_precision"] - s020["person_precision"] <= REQ["precision_drop"]),
        ("miou_drop", BASE["miou"] - s020["miou"], REQ["miou_drop"],
         BASE["miou"] - s020["miou"] <= REQ["miou_drop"]),
    ]
    finite = all(math.isfinite(v) for r in (s020, s002) for v in r.values() if isinstance(v, float))
    checks.append(("no_nan_or_contract_failure", 1.0 if finite else 0.0, 1.0, finite))
    rows.append({"epoch": epoch, "s020": s020, "s002": s002,
                 "checks": [{"criterion": c, "value": v, "threshold": t, "ok": bool(o)}
                            for c, v, t, o in checks],
                 "passed": all(o for _, _, _, o in checks),
                 "mean_f1_s020": 0.5 * (s020["vehicle_f1"] + s020["person_f1"])})

passing = [r for r in rows if r["passed"]]
selected = max(passing, key=lambda r: r["mean_f1_s020"]) if passing else None
result = {"gate": "disc_lr_v1", "canonical_baseline": BASE, "requirements": REQ,
          "epochs": rows,
          "verdict": ("DISCRIMINATIVE_LR_GATE_PASSED_PENDING_REVIEW" if passing
                      else "DISCRIMINATIVE_LR_NO_GAIN_HYBRID_CLOSED"),
          "selected_epoch": selected["epoch"] if selected else None,
          "selected_checkpoint": selected["s020"]["checkpoint"] if selected else None,
          "selected_checkpoint_sha256": selected["s020"]["checkpoint_sha256"] if selected else None,
          "selection_rule": "both epochs gated individually; if both pass, higher mean vehicle/person F1 at score 0.20"}
(EXP / "gate_disc_lr_v1.json").write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, indent=2, sort_keys=True))
PYEOF

VERDICT=$("$PY" -c "import json;print(json.load(open('$EXP/gate_disc_lr_v1.json'))['verdict'])" 2>/dev/null || echo IMPLEMENTATION_BLOCKED)
finish "$VERDICT"
