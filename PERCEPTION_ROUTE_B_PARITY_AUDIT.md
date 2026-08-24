# Route B perception parity audit — forensic result

**Date:** 2026-08-24
**Scope:** bounded, parity-first. Read-only on all existing evidence. No CARLA, no OAI, no collection,
no training, no checkpoint writes, no production-code edits. Route B accuracy numbers were **not** recomputed.
**Artifacts:** `experiments/route_b_perception_parity_audit_v1/20260824_163730_EDT/`

---

## 1. Terminal decision

> ## `ROUTE_B_INPUT_PARITY_FAILED_REPLAY_PENDING`

Static evidence proves **material, model-input-affecting non-parity** between the historical M-prime
collection and the Route B 30/30 collection, and the historical replay bundle is **unavailable** on this
machine, so the exact replay gate could not be executed.

The previous `CLEAR_ROUTE_B_COVERAGE_DEGRADATION` terminal is **retired**. The radar cadence/window mismatch
flagged in the audit request is not merely unrefuted — it is confirmed from both sides with retained
per-frame evidence, and a second, independent radar mismatch (halved per-sweep detection count) was found
that was not previously known.

---

## 2. Was historical replay possible? No.

`fusion_training_data/moving_ego_pps200000_merged_8loops_stride2` **does not exist**. Both
`gate_eval/dataset` symlinks resolve by path but the target is absent — exactly the dangling-link trap the
audit request warned about. `readlink -f` returns a clean path; `stat -L` fails with `No such file or
directory`. No `manifest.csv`, no `object_boxes.csv`, no `rgb/`, `masks/`, `radar_tensors/` or
`radar_points/`. No tarball, archive or rsync backup of it exists anywhere on the filesystem.

Because noAE could not be replayed, **AE64 was not run and no Route B result was reinterpreted**, per the
stop rule.

`HISTORICAL_REPLAY_BUNDLE_REQUEST.txt` has been produced. It names **96 frozen sample IDs**, 32 from each
of the three source densities, ordered deterministically by `(priority tier, sha256(sample_id))`. All 96
land in **tier 1**: every one is present in the retained clean noAE per-object rows *and* carries both
vehicle and person ground truth. No substitution was needed.

### What survived, and what it bought us

Two retained artifacts carry historical **per-frame** evidence and made this audit conclusive despite the
missing dataset:

| Artifact | What it is | What it proves |
|---|---|---|
| `staleness/egospeed_split_ds/manifest.csv` | A **real 34 MB file copy** (not a symlink) of the historical collection manifest — 15,183 rows across the three source collections, same 51 columns as Route B | Historical per-frame radar detection counts, saved-frame timing, camera intrinsics and pose |
| `experiments/ae_integrated_20260710/*/gate_eval/metrics/test_learned_object_metrics.csv` | Retained clean gate per-object rows (noAE and AE64) | Historical eligible-GT composition; test-frame identity (1,703 sample IDs, **all** present in the manifest copy) |

Its sibling `rgb/`, `masks/`, `radar_tensors/`, `radar_points/` and `object_boxes.csv` entries are dangling
symlinks, so it supplies **metadata only** — no images, no tensors. Note also that its `split` column is the
per-source collection split (14,238/632/313), **not** the merged 72/14/14 re-split that produced the
2,162-frame gate test set; historical test-frame identity was therefore taken from the retained clean rows,
never from that column.

---

## 3. Confirmed material mismatches

Four of nineteen parity rows are `MISMATCH`, and **all four are radar-input rows that feed the model
directly**. Full detail in `parity_matrix.csv`; measured distributions in `input_distribution_summary.csv`.

### 3.1 Radar temporal window: 2 sweeps over 0.2 s → 1 sweep over 0 s

The historical collector builds the deque **after** stride rejection
(`carla_collect_moving_ego_fusion_training_data.py:1001` `continue` precedes `:1017` `build_radar_sample`
and `:1031` `append`), so it holds the last two **saved** tensors and takes their per-channel maximum.
Recipe value 2 is corroborated by `scripts/run_moving_ego_fusion_training_pipeline.sh:17` (default 2) and
`scripts/run_pps_ablation.sh:60` (passes 2 explicitly for the pps200000 collections). At 10 Hz with
stride 2 the two inputs to the max are **0.200 s apart**.

Route B sets `radar_temporal_window_frames=1`
([run_route_b_perception_collection.py:245](data_collection/run_route_b_perception_collection.py#L245)) — no
deque, no max, a single sweep.

Corroboration that window=1 is the outlier, not the norm: every other pipeline in the repo passes 2 —
`data_collection/phase2_paired_causal_collector.py:152`, `run_phase2_calibration_audit.py:1216`,
`configs/policy_corpus_advisor_rich_v3.yaml:65`, and the retained resolved config
`staleness/uplink_only_latency_budget/.../fusion_ego_4608_resolved_config.json:111`.

### 3.2 Per-sweep radar detection count is halved — a second, independent mismatch

This was not previously known and is measured, not inferred:

| | Historical | Route B | Ratio |
|---|---|---|---|
| radar detections per consumed sweep (mean) | **18,624** | **9,293** | 0.499× |
| median | 18,767 | 9,360 | |
| **maximum** | **20,000** | **10,000** | |

Those maxima are exact: `200,000 pps × 0.1 s = 20,000` and `200,000 pps × 0.05 s = 10,000`. The requested
`sensor_tick` of 0.1 s **did not** produce a 0.1 s radar integration window under Route B's 20 Hz simulator
step; the observed integration window is the 0.05 s simulator step. Configured radar attributes are
identical on both sides (row 4 is a `MATCH`) — it is the *realized* sweep that differs.

**Combined effect of 3.1 and 3.2:** the historical model input rasterized the union of two sweeps of
~18,600 detections each (~37,200 detections); Route B rasterizes one sweep of ~9,300. That is up to **~4×
fewer radar detections** entering the four radar channels, which are 4 of the 7 fused input channels.

The resulting difference in occupied-pixel fraction is **not measured and is not claimed**, because no
historical radar tensor survives. Route B's own measured occupancy is recorded for the bundle comparison:
occupancy channel nonzero fraction 0.246 (median 0.251).

### 3.3 Radar processing cadence: 5 Hz → 2 Hz

`build_radar_sample` runs once per saved sample in both collectors, but "saved" means different things:
historical **0.2000 s** (frame_id step exactly 2, n=15,180 intervals), Route B **0.5000 s** (frame_id step
exactly 10, all 598 intervals). Historically 1 radar sweep in 2 was consumed; in Route B, 1 in 10.

### 3.4 Stationary-track accumulator: same parameters, different discrete-time behaviour

`StationaryTrackAccumulator` is the same class with the same four values on both sides (0.35 m/s, 5.0 s,
1.5 m grid, 2.0 s stale), but `update()` is driven at 5 Hz versus 2 Hz. Per-update `dt` is 0.2 s versus
0.5 s; the 2.0 s stale horizon tolerates 10 missed updates versus 4; and the ego travels ~3.5 m between
updates at 25 kph versus ~1.4 m historically, against a 1.5 m association grid. Route B's `stationary_age`
channel is entirely zero on the median frame (nonzero-pixel fraction median 0.000, p90 0.250).

The retained manifests show the stationary *fraction* is similar (34.0% historical vs 31.1% Route B of all
detections) while the absolute count is halved — so 3.2, not the tracker parameters, dominates here.

### 3.5 Two rows remain `UNKNOWN`

- **Weather.** Neither collector *sets* weather; both inherit the fresh-world default. But the historical
  collector never *recorded* it (`write_moving_metadata` has no weather field), so parity cannot be closed
  from retained evidence. Route B RGB is not anomalously dark despite the recorded
  `sun_altitude_angle: 0.0` — mean intensity 104.6/255, dark-clipped pixel fraction 0.027 — so a gross
  exposure difference is not indicated.
- **Renderer quality level.** Unrecorded on **both** sides. `CLAUDE.md` already flags the M-prime training
  renderer as unrecorded and forbids retroactively relabelling it; this audit does not.

Only the replay bundle's RGB frames could settle either row, and then only indirectly.

---

## 4. What was checked and found clean

Thirteen rows are `MATCH`. Two deserve explicit mention because they were live hypotheses.

### 4.1 The Route B evaluator is faithful — no discrepancy found

[run_route_b_eval_pilot_v1.py](experiments/route_b_30_30_perception_pilot_20260824/run_route_b_eval_pilot_v1.py)
**imports** `load_fused_tensor`, `load_mask`, `update_confusion`, `decode_objects`,
`greedy_match_predictions`, `load_object_boxes`, `valid_localization_objects` and
`build_multitask_fusion_lraspp` directly from the official modules. Its one reimplemented function,
`build_model`, matches `evaluate_fusion.py:210-249` field for field, including the integrated feature-AE
attach from `ckpt['trial']['ae_bottleneck']` **before** `load_state_dict`. The prediction range filter is
the same expression on both sides. The segmentation path is the same four statements. Every decoder setting
matches the retained historical command lines in `rl_agent/ae_integrated/run_noae_baseline.sh` and
`run_ae_integrated.sh` (score 0.20, NMS radius 2 px, top-k 120, class-aware 5 m match, ≤40 m range,
`min_gt_area_px` 12.0 from `fusion_full_run.yaml`). Both checkpoint SHA-256 digests verify against the
registered expected values.

This is a **code-reading verdict, not a replay verdict.** It cannot substitute for the numeric replay gate.

### 4.2 Person GT is correct — do not chase this

Route B uses the identical `rasterize_person_regions(mask, boxes, shape="box")` call on the identical
`build_object_rows` output. The documented convention holds: vehicle IoU is semantic silhouette overlap,
person IoU is projected-box-region overlap, and detection/localization is world-coordinate based.

There is a trap in the manifest columns. **`person_pixels` is 0 on all 15,183 historical rows** — because
the historical collector calls `build_manifest_row` *before* painting the person boxes and only then
rewrites the mask PNG (`carla_collect_moving_ego_fusion_training_data.py:1100-1116`). Route B recomputes
both pixel columns *after* painting. This is **metadata-only ordering**, not missing or different person GT;
the historical person IoU of 0.590 confirms the historical masks on disk do contain person regions.
`vehicle_pixels` is likewise pre-paint historically and post-paint in Route B.

### 4.3 Scene difficulty does not explain the gap

Eligible-GT composition is close, and where it differs Route B is **easier**:

| | Historical | Route B |
|---|---|---|
| eligible person GT per frame | 0.852 | 0.836 |
| eligible vehicle GT per frame | 1.470 | 1.529 |
| median person box area (px) | 344 | 557 |
| median vehicle box area (px) | 4,997 | 5,385 |
| median vehicle GT distance (m) | 25.5 | 15.6 |
| median person GT distance (m) | 29.4 | 24.4 |

Camera geometry also matches: identical intrinsics (fx 369.504172, cx 640.0, FOV 120°) and closely
overlapping realized pose (pitch −4.21° vs −4.29°, z 1.527 m vs 1.535 m).

---

## 5. What the existing Route B results can and cannot support

The Route B numbers (noAE vehicle recall 0.514 / person recall 0.168; AE64 0.599 / 0.226; degraded
localization and segmentation) were not recomputed and are not disputed as *measurements*.

**They can support:**
- A valid measurement of how these two checkpoints perform **on Route B inputs as actually collected**.
- The conclusion that collision windows do not explain the degradation (already established; unchanged).
- The conclusion that eligible-GT composition, camera geometry, GT convention, decoder settings and the
  evaluator do not explain it (established here).

**They cannot currently support:**
- Any claim that the model fails to generalize to Route B geography. The radar input distribution is
  materially different from the one the model was trained on, so this is confounded — a change of route
  *and* a change of sensor input, measured together.
- Any comparison against the historical metrics as a like-for-like baseline.
- Any authorization to retrain, retune, or revise the perception model.

**Independent limitation, unchanged by this audit:** the historical test split is 2,162 hash-selected frames
from repeated laps of the same ~268.7 m route, all test spatial cells overlap training cells, and ~44.8% of
adjacent saved-frame pairs cross split boundaries. Those metrics are **same-route, in-distribution
performance** — not proof of full-map generalization. Even after parity is restored, a single Route B
episode does not constitute independent validation.

---

## 6. Single smallest next action

> **Request the 96-ID historical replay bundle** specified in
> `experiments/route_b_perception_parity_audit_v1/20260824_163730_EDT/HISTORICAL_REPLAY_BUNDLE_REQUEST.txt`
> from whichever machine or backup still holds
> `fusion_training_data/moving_ego_pps200000_merged_8loops_stride2`.

Nothing else should start first. Until noAE reproduces the retained clean rows on those exact IDs, we cannot
distinguish "the code/checkpoint no longer reproduces history" from "Route B inputs are different," and a
parity-correct recollection would be built on an unverified reference.

The merged `metadata.json` is the single highest-value item in the bundle: its embedded `command_args`
records the exact historical `--fps`, `--sample-stride`, `--radar-temporal-window-frames` and
`--radar-points-per-second`. This audit inferred those from code, retained pipeline scripts and measured
per-frame evidence; `command_args` would make them primary.

---

## 7. Minimal parity-correct collection design (specification only — not implemented, not run)

Authorization for this is **not** requested here; it is blocked behind the replay gate above.

### 7.1 Why naïvely maxing two 2 Hz saved tensors is invalid

The obvious "fix" — set `radar_temporal_window_frames=2` in the Route B collector — is **wrong**, and would
produce a third input distribution matching neither side.

With Route B's current save path, `build_radar_sample` is only called inside `save_frame`, i.e. every 10th
simulator tick. A deque of the last two *saved* tensors therefore combines sweeps **0.5 s apart**, not
0.2 s. At the qualified 25 kph the ego travels ~3.5 m in that interval, and a crossing pedestrian at
1.4 m/s travels ~0.7 m. Taking a per-channel maximum over that gap does not densify a single observation —
it **smears two spatially distinct observations** into one tensor, producing doubled ghost returns
displaced by metres in the image plane. The historical 0.2 s union smears by ~1.4 m of ego motion; a 0.5 s
union smears by ~3.5 m. It would also do nothing about the halved per-sweep count (§3.2), so the total
detections rasterized would still be ~18,600 against the historical ~37,200 — and now misregistered.

Same knob value, different physical semantics. Parity is a property of the **effective temporal span and
the realized detection count**, not of the integer in the config.

### 7.2 What parity actually requires

| Requirement | Target | Why |
|---|---|---|
| Radar integration window per sweep | **0.1 s** (≈20,000 detections at 200,000 pps) | Restores §3.2. Requires the radar to actually integrate over 0.1 s, which under the current build means a 0.1 s simulator step — verify by asserting the realized `radar_points` ceiling is 20,000, not by trusting `sensor_tick`. |
| `build_radar_sample` / tracker call cadence | **5 Hz (every 0.2 s)** | Restores §3.3 and §3.4. Must be decoupled from the archive-write cadence. |
| Temporal window | **2 tensors, effective span 0.2 s** | Restores §3.1. |
| Saved-frame cadence | free (2 Hz is fine) | Storage is not a parity variable. |

The key structural change is that **radar processing cadence must be decoupled from save cadence**. The
historical collector conflated them and got 0.2 s by coincidence of `fps=10, stride=2`. A parity-correct
Route B collector must call `build_radar_sample` (and therefore `tracker.update`) every 0.2 s regardless of
whether that frame is archived, push each result into the 2-deep deque, and archive only the frames it
wants — at 2 Hz or any other rate. Reusing the existing save-gated call site cannot produce a 0.2 s span at
a 2 Hz save rate.

Every run should assert and record, as fail-fast preconditions: realized `radar_points` ceiling = 20,000;
consecutive `build_radar_sample` call spacing = 0.200 s; `radar_temporal_window_frames` = 2; and the
renderer quality level, which neither side currently records (§3.5).

---

## 8. Deliverables

`experiments/route_b_perception_parity_audit_v1/20260824_163730_EDT/`

| File | Contents |
|---|---|
| `parity_matrix.csv` | 19 rows × 9 columns. Per row: historical evidence source, Route B evidence source, both values, MATCH/MISMATCH/UNKNOWN, whether it can change model input / GT / metadata only, and likely impact. 13 MATCH, 4 MISMATCH, 2 UNKNOWN. |
| `input_distribution_summary.csv` | Descriptive statistics (n, mean, median, p10, p90, min, max) for both sides where available. Historical RGB and radar-tensor rows are explicitly `UNAVAILABLE` rather than substituted. |
| `HISTORICAL_REPLAY_BUNDLE_REQUEST.txt` | The 96 frozen sample IDs with tier annotation, the deterministic selection rule, the exact per-ID file list, and the replay procedure that will be run on receipt. |
| `route_b_parity_audit_v1.py` | The create-only audit script that generated the two CSVs and the bundle request. Offline; no CARLA, no OAI, no inference, no checkpoint access; writes only into its own directory. |

`historical_replay_comparison.csv` was **not** created — replay was not possible, and an empty or
substituted file would misrepresent the evidence.

No existing experiment evidence was modified, overwritten or deleted. No Route B metric was recomputed.
CARLA is down and was never started.
