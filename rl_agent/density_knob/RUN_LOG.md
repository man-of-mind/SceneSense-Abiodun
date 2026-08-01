# Density-adaptive knob selection — run log

Executes `rl_agent/DENSITY_ADAPTIVE_KNOB_PLAN.md`. Session date **2026-07-31**.
Home: `rl_agent/density_knob/`. Results: [`DENSITY_KNOB_RESULTS.md`](DENSITY_KNOB_RESULTS.md).

## Machine state at start (guardrail 6)
`/proc/loadavg = 8.33`; GPU 60% util, 10.3/32.6 GB used. **Another session's work was already running
and was left alone**: CARLA (pid 1742137, rpc 2000), OAI gNB + nr-UE (rfsim, T-tracer on 2021/2023),
and a `fusion-back` container running `carla_fusion_staleness_scenario.py --role back` (the pending
uplink-only-over-OAI run). Nothing was killed and no second CARLA was launched.

**Consequence for this analysis:** it was run **entirely offline on the GPU** (no CARLA client, no
socket, no OAI), so it could not disturb that run. That was possible because the whole question is
answerable by re-running the offline per-model eval on existing captures — see "reuse" below.

## What was reused vs re-measured

**Reused as-is (no re-measurement):**
- dataset / capture: `fusion_training_data/moving_ego_pps200000_merged_8loops_stride2` (test split,
  2162 frames) — the same corrected-drivable moving-ego capture the M' knob matrix was built on.
- the four integrated-AE M' checkpoints (`noae`, `ae32`, `ae64`, `ae128`) under
  `experiments/ae_integrated_20260710/`.
- all matching / decoding / GT / codec code: `object_targets.decode_objects`,
  `greedy_match_predictions`, `valid_localization_objects`, `split_runtime`,
  `carla_split_inference_udp_data_collect.TransportConfig`.
- eval knobs identical to `run_zstd_full_overnight.sh`: score 0.20 / nms 2 px / topk 120 /
  match 5.0 m / max-GT 40 m / zstd level 3.
- `loopback_latency_zstd.json` (36 measured ideal-loopback profiles) for the payload→uplink-ms fit.
- `PERMODEL_KNOB_MATRIX_ZSTD.md` as the reproduction target and the 0.95 m accuracy anchor.
- live GT CSVs from `staleness/uplink_only_latency_budget/fresh_run_20260730_000257` — used **only**
  to measure the origin-vs-bbox-centre convention delta (they carry both columns).

**Re-measured (the new work):**
- `density_knob_eval.py` — a new offline driver that emits **one row per (profile × frame)** with
  payload bytes, in-view GT count, and per-class tp/fp/fn/loc. The packaged `evaluate_fusion` only
  emits an aggregate `payload_bytes_mean`, so payload could not be joined to a frame's density.
- **72 profiles** = 4 AE {none,32,64,128} × 3 quant {u8,u6,u4} × **6 ROI q {0, 0.3, 0.5, 0.7, 0.9,
  0.98}** × 2162 frames = 155 664 profile-frames. The published matrix only had q ≤ 0.5; the plan
  needs high q (τ→1.0), so **q = 0.7 / 0.9 / 0.98 are new measurements.**
- `build_frame_density.py` — post-hoc per-frame density label + confound controls.

## Pipeline as run

```bash
AB=.../neu_collab/abiodun; cd $AB
export PYTHONPATH="$AB/pole_lraspp_multimodal_fusion:$AB:$AB/rl_agent/feature_ae"   # offline eval only
export MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen CUDA_VISIBLE_DEVICES=0

python3 rl_agent/density_knob/build_frame_density.py                 # 2162 test frames -> density label
python3 rl_agent/density_knob/density_knob_eval.py \
    --models noae,ae32,ae64,ae128 --quants uint8,uint6,uint4 \
    --rois 0.0,0.3,0.5,0.7,0.9,0.98 --out-dir rl_agent/density_knob/raw     # ~55 min GPU
python3 rl_agent/density_knob/gate_density_eval.py                   # G1..G6, must pass first
python3 rl_agent/density_knob/analyze_density_knob.py                # tables + Pareto + plots
```

`PYTHONPATH` **is** exported here and that is safe: these are offline eval/analysis scripts that
never construct a CARLA client. No front/back/loopback client was started at any point in this
session, so the `UDPMessageSocket ... unexpected keyword 'remote_host'` shadowing trap
(memory `dont_set_pythonpath_for_carla_client`) cannot apply.

## Artefacts

| file | what |
|---|---|
| `DENSITY_KNOB_RESULTS.md` | the write-up: tables, lookup, plots, caveats |
| `raw/perframe_{noae,ae32,ae64,ae128}.csv` | per (profile × frame) rows — the primary data |
| `raw/frame_density.csv` | post-hoc density label + confounds per test frame |
| `raw/by_density_profile.csv` | (profile × density bin) aggregate — payload/recall/loc/FP |
| `raw/payload_vs_density.csv` | payload per bin per profile + spread % (the physics check) |
| `raw/best_knob_lookup.csv` | density → best-knob policy table |
| `raw/tables.md` | auto-generated markdown tables (results doc never hand-transcribes) |
| `raw/gate_report.txt` | G1–G6 pass/fail with the numbers behind each |
| `raw/eval_settings.json`, `raw/analysis_settings.json` | exact knobs + tolerances + latency fit |
| `plots/*.png` | Pareto per bin, cost-of-ROI-drop, payload spread |
| `density_eval.log` | the 55-min eval log |

## Machine state at end
Load average 4.33 (down from 8.33). The **OAI gNB + nr-UE and the `fusion-back` container are still alive
with their original PIDs** (1932618 / 1933679 / 1934354) — untouched. The CARLA server that was up at the
start (pid 1742137) is **no longer running**; it was not this session's to manage and nothing here touched
it. My scripts contain no `kill`/`pkill`/`subprocess` calls at all (verified by grep) and never opened a
CARLA client — the whole analysis was GPU PyTorch + CSV. `/tmp/carla_roadstate.log` was last written at
09:12, i.e. **hours after this eval finished at 02:54**, so another session was operating CARLA in that
window. Flagging it rather than asserting a cause.

## Timing
`build_frame_density.py` ~5 s · `density_knob_eval.py` **54 min** GPU (noae 15.9, ae32 12.4, ae64 11.5,
ae128 12.6 min; 2162 frames × 18 profiles each) · gates ~20 s · analysis + plots ~15 s.

---

## Addendum — SEG-INCLUSIVE RE-RUN (2026-07-31, same day)

**Trigger.** Supervisor/colleague review: the first run scored **detection only**. The shared map also
needs the **segmentation** layer (drivable surface / lane / dense vehicle-person masks between objects), so
the density policy had to be re-derived with seg in the objective.

**What was added (not re-collected):**
- `density_knob_eval.py` now also evaluates the seg head — `outputs["out"]` (already produced, previously
  discarded) → interp to full res → argmax → per-frame 3×3 confusion (bg/vehicle/person), emitted as
  `conf_00..conf_22`. mIoU + per-class IoU bin by density with a post-hoc SUM, exactly as the matrix does.
- `gate_density_eval.py`: new **G7** — the seg columns must reproduce the published matrix mIoU + veh IoU
  on the ROI 0/0.3/0.5 profiles (anti-bug check for the seg head). Result: exact (noae/u8/ROI0 mIoU
  0.840=0.840, veh IoU 0.931=0.931).
- `analyze_density_knob.py`: Pareto accept is now **JOINT** (detection recall/loc AND seg mIoU within 2 pts
  of the model's own ROI-0 seg in the same bin). Reports both the detection-only pick and the seg-aware
  pick; new table T3b (seg-vs-q collapse) and plot `seg_collapse_vs_roi.png`.

**Re-run.** Fully offline GPU again (the CARLA server was killed mid-session by another user — irrelevant,
this eval never touches CARLA). All 72 profiles re-evaluated (~57 min). Detection numbers reproduce the
first run exactly (determinism check) and G2 still reproduces the matrix; seg is new. **9/9 gates PASS.**
The detection-only raw CSVs were moved to `raw/detonly_backup/`; `raw/perframe_*.csv` now carry seg.

**Result (flips the headline).** With seg required, the ROI knob is unusable — it destroys seg at every
density (veh IoU 0.92→0.11 as q 0→0.98). The seg-aware policy is **`ae32/u4/ROI 0`, ~90 KB, IDENTICAL at
every non-empty density** (bins 1-2 / 3-4 / 5+ all pick it). Density adaptation via ROI was an artifact of
ignoring seg. ROI-0 latency is already **measured** (~39 ms). See `DENSITY_KNOB_RESULTS.md` §1–§3.

**High-ROI latency sweep — DONE 2026-07-31 19:17–19:26** (`run_ideal_loopback_zstd_highroi.sh`, 12 profiles
q=0.7/0.9/0.98, reused the restarted CARLA, ~45 s/profile, 12/12 OK, delivery 1.00). Merged into
`loopback_latency_zstd.json` (36→48 profiles). Confirms the earlier extrapolation: front flat 24.5–26 ms
(AE profiles), transport 1.3–4.1 ms; fit `1.260 + 0.00877·kB` ≈ the old `1.067 + 0.00912`. **No policy
pick changed** — latency was never the deciding axis. The whole ROI range is now measured, not derived.
