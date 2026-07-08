#!/usr/bin/env bash
AB=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
cd "$AB"; source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate 2>/dev/null
LOG="$AB/rl_agent/analysis/accuracy_run.log"
echo "[$(date '+%F %T')] waiting for GPU (zstd sweep to finish)..." >> "$LOG"
while pgrep -f "sweep_runner.py" >/dev/null 2>&1; do sleep 30; done
echo "[$(date '+%F %T')] GPU free -> re-aggregate 8-profile payload sweep, then accuracy validation" >> "$LOG"
python3 rl_agent/sweep_analyze.py >> "$LOG" 2>&1 || true
echo "[$(date '+%F %T')] accuracy VALIDATION (baseline + uint8)" >> "$LOG"
bash rl_agent/run_accuracy_sweep.sh baseline q_pchan_u8_zlib >> "$LOG" 2>&1
python3 - >> "$LOG" 2>&1 <<'PY'
import json
from pathlib import Path
S = Path("experiments/rl_accuracy_sweep")
def miou(n):
    f = S/n/"metrics"/"test_fusion_evaluation_metrics.json"
    return json.load(open(f)).get("miou") if f.exists() else None
b, u = miou("baseline"), miou("q_pchan_u8_zlib")
ok = (b is not None and 0.78 <= b <= 0.90 and u is not None and abs(u-b) <= 0.05)
print(f"VALIDATION baseline_miou={b} uint8_miou={u} (ref baseline=0.837) -> {'PASS' if ok else 'FAIL'}")
Path("/tmp/acc_validation").write_text("PASS" if ok else "FAIL")
PY
if [ "$(cat /tmp/acc_validation 2>/dev/null)" = "PASS" ]; then
  echo "[$(date '+%F %T')] validation PASS -> running remaining profiles" >> "$LOG"
  bash rl_agent/run_accuracy_sweep.sh q_pchan_u4_zlib q_ptensor_u8_zlib q_pchan_u8_none q_pchan_u4_none q_ptensor_u8_none >> "$LOG" 2>&1
  echo "[$(date '+%F %T')] ACCURACY SWEEP DONE -> rl_agent/analysis/accuracy_vs_compression.md" >> "$LOG"
else
  echo "[$(date '+%F %T')] VALIDATION FAILED -> codec injection likely wrong; NOT running full sweep. Needs review." >> "$LOG"
fi
