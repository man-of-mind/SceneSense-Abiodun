# Data-collection plan — richer CARLA corpus to build the environment cleanly

**Why:** the current replay corpus (staleness study) is the binding limit on the policy environment. Inspection
(2026-08-10) confirmed the traces are physically sound BUT: (a) **ground truth is vehicles only — no pedestrian
GT**, so pedestrian localization/recall cannot be scored at all; (b) the shield selected a SPLIT schedule on
5.83% of control ticks and admitted only 15 matched-object sends, so safety denominators are thin; (c) observation
coverage is 45.18%. Separately, the observation-based predicate defined below marked a send as needed on 14.66%
of the three current replay trajectories. We collect a purpose-built corpus that fixes these limits so the
environment covers the safety-critical cases it is currently blind to. Real CARLA data is the primary path
(internship extended +3 months → do it properly, not a synthetic stub).

**Scope discipline:** this is a *data* task, not a new pipeline. Reuse the existing staleness collector; background
pedestrian spawning already exists, so the only perception-path delta is logging pedestrian ground truth. Scenario
arguments, batch orchestration, replay class preservation, and verification are support work. Do NOT expand beyond the scenarios
below without a note here.

## 1. Reuse, don't rebuild
- Base collector: `uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py` (produces the
  exact `*_object_ground_truth.csv` + `*_object_predictions.csv` schema the surrogate already parses).
- Pedestrian-spawn reference (already working in-repo): `radar_camera_lidar_data_collect_update_pedestrian_
  vizualizor_fusion.py` and `carla_collect_moving_ego_fusion_training_data.py`.
- Keep the shared collector unchanged. Put a thin derived entry point in `abiodun/data_collection/` that delegates
  to the shared collector and replaces only its GT-row builder. This avoids a divergent ~6,500-line copy while
  preserving the "never edit the shared original" constraint.

## 2. The only functional change: log pedestrian ground truth
The detector already emits `person` predictions (confirmed in the current prediction CSVs), so perception needs
no change. The gap is purely on the GT side:
- Use the existing controllable **walker** count and CARLA walker AI. Requested spawn counts are inputs, not proof
  of usable data: a smoke run must confirm walkers actually enter the ego field of view and ≤25 m; adjust only the
  pre-registered scenario arguments if this realization gate fails.
- Log pedestrian actors into `*_object_ground_truth.csv` with the **same columns and the actor-origin position
  convention** already used for vehicles (`origin_x/origin_y`, `class_name=pedestrian`, `distance_m`,
  `in_camera_frustum`, size fields). Do not switch to bbox-center (reintroduces the known 1–1.3 m bias).
- Everything else (frame cadence ~8–10 fps, ego autopilot, vehicle spawns, streams layout) stays as in the base
  collector.

## 3. Scenarios and locked split (24 runs, 500 processed frames each at 10 Hz)
Three scenario archetypes, eight independent CARLA trajectories each. The batch config pre-registers unique seeds
and assigns **4 train / 2 validation / 2 test trajectories per archetype** (12/6/6 total). Splitting is by whole
trajectory, stratified by archetype, never by frame; seeds are disjoint across splits. The explicit split manifest
is authoritative—do not let the registry hash broad names such as `ped_crossing` into one split.

The three archetypes are:
1. **`ped_crossing`** — pedestrians (5–15) crossing/walking near the ego path, moderate vehicle traffic. The
   critical missing class. Ensures a real pedestrian-recall + pedestrian-localization denominator.
2. **`dense_fast`** — deliberately balanced as two pre-registered subprofiles because maximum density and maximum
   speed conflict in urban traffic: four **dense-flow** runs raise vehicle count, while four persistent
   **fast-convoy** runs hold a lead in view at target 30–45 mph. Each split contains both variants. Together they
   expand density and high-speed localization-under-latency without pretending a gridlock run achieved 50 mph.
3. **`mixed_urban`** — balanced veh + ped at nominal density/speed, the "typical operating point" for headline
   numbers.
Aim for meaningfully higher **observation coverage** than the current vehicle-only 45.18% by keeping actors in-FOV
and ≤25 m; report object-row and frame-level coverage per class and run. Spawn requests and target speeds are not
achieved conditions; the verifier reports realized densities and finite-difference speeds. Compare the legacy
45.18% only against **vehicle** replay coverage after the same interpolation/track-hold logic. Pedestrian replay
coverage has no honest legacy denominator, so require non-zero truth and observed pedestrian object-frames and
report its coverage separately. Direct same-frame match coverage is a separate, stricter diagnostic and must not
be compared to the held-track replay baseline.

## 4. Machine + safety rules (from CLAUDE.md — do not violate)
- **Collect on codex's box (L10319), which also has CARLA 0.10.0.** Rationale: codex runs the whole downstream
  chain (environment → controllers → evaluation) there, so collecting on the same CARLA/same box keeps one
  provenance chain and removes any cross-version drift (build/map-asset/detector differences) between where data
  is made and where it is consumed. codex owns the full loop end-to-end.
- **Prerequisite check before collecting (confirm on L10319):** (a) the front fusion **detector weights are
  present and the perception pipeline runs** — needed to emit `_object_predictions.csv`, not just the surrogate;
  (b) same CARLA **0.10.0 shipping build + Town10HD_Opt** assets as prior traces; (c) GPU is free enough to avoid
  the render-throttle bug (§ below).
- **Do NOT export `PYTHONPATH`** for the CARLA client (shadows `abiodun/` with the stale `neu_collab/` copy →
  the `UDPMessageSocket … remote_host` failure).
- **GPU placement on L10319:** only one RTX 5090 is visible, so a physically separate back-half GPU is impossible.
  Preflight both reasonable placements in short runs and lock the one that keeps rendering healthy; prefer placing
  the back half on CPU if that avoids concurrent tail/render contention. Record the resolved devices and GPU
  inventory. Watch `camera_frame_wait` near the healthy ~32 ms regime, never the prior ~122 ms throttle regime.
- Check `/proc/loadavg` + `docker ps` first; **reuse** a running CARLA server; never kill another user's
  CARLA/OAI. Town10HD_Opt for consistency with prior work.
- This is uplink/collection only — no OAI closed-loop needed for the corpus itself.

## 5. Original pre-registered verification gates (historical contract)
These gates remain here unchanged as the contract under which candidate v1 was collected and first graded.
After Abiodun clarified that phase 1 is fresh-map control for all in-scope objects, gate 4's shield-trajectory
`send_needed` comparison was found to be mis-specified for the corpus-motion question. The immutable original
verification remains valid as a record of that test; §9 documents the versioned freshness re-score that
supersedes only the salvage/top-up disposition. The other collection-health and detection diagnostics remain
useful evidence.

Emit a `CORPUS_VERIFICATION.md` per collection batch with:
1. **Pedestrian GT present:** every `ped_crossing`/`mixed_urban` run has in-frustum, ≤25 m
   `class_name=pedestrian` rows. Allow stopped/starting walkers: all finite-difference speeds must be plausible
   (0–3.5 m/s), and separately report the active-mover (`speed>0.2 m/s`) median and percentiles. Use broad CARLA
   actor plausibility bounds (height 0.8–2.2 m, width 0.2–1.0 m) and report the realized distribution rather than
   rejecting valid child blueprints against an adult-only target range.
2. **Position sanity:** `origin_x/origin_y` within town bounds; `distance_m` ≥ 0; no NaN explosions in GT
   positions; `in_camera_frustum` ∈ {0,1}.
3. **GT↔prediction matchability:** run the existing staleness matcher (actor-origin, score ≥0.20, 5 m greedy
   one-to-one gate) and report per-class **object-row coverage** plus **frame coverage**. `person`, `pedestrian`,
   and `walker` normalize to `pedestrian`; matched class labels must remain pedestrian in replay.
4. **Send-path exercised:** run ε=2 m/core90/range25 with the locked shield and define
   `send_needed = (SKIP not in raw_safe_action_ids) AND (at least one SPLIT action is raw-safe)`. Report this
   independently from selected `split_pct` and actual `capture_attempt_pct`. The current three-replay baseline is
   14.66% send-needed and 5.83% selected-SPLIT. The corpus gate is send-needed >14.66% overall, selected SPLIT
   >5.83% overall, and send-needed >14.66% in the `fast_convoy` variant specifically; report every family/variant
   so the dense-flow half cannot mask a high-speed scenario that fails to exercise the path.
5. **Speed/density achieved:** report per-family actor counts + speed distributions vs the targets in §3.
6. **Pipeline/timing health:** every run has non-empty predictions and GT, at least 95% of requested processed
   frames, no result-receive collapse, and healthy `camera_frame_wait`: median ≤60 ms and p95 ≤100 ms. Also report
   the measured median/p95 rather than only PASS/FAIL.

Any failed gate quarantines the batch. Do not silently drop failed runs and relabel the remainder; amend the batch
manifest and collect a new sibling batch or documented replacement run.

## 6. After collection
1. Point the surrogate `replay.roots` and explicit split manifest at the immutable verified batch (keep the old
   vehicle-only set as a labelled legacy comparison).
2. Re-run the ε2/core90/range25 pilot on the new corpus → confirm pedestrian metrics now have real denominators
   and the shield's safety numbers are no longer denominator-starved.
3. THEN proceed to the controller ladder (rule → bandit → MPC → RL) on the clean environment.

## 7. Open questions for Abiodun / advisor
- Pedestrian-recall **hard floor** value (safety-critical class) — still advisor-pending; this corpus is what
  makes it measurable.
- Confirm 25 m as the headline range (data already favors it; 40 m stays diagnostic).
- Who runs CARLA: **decided — codex runs the full loop on L10319** (its CARLA 0.10.0), so collection and
  consumption share one box/version. local Claude reviews the plan + verification output.

## 8. Original execution outcome — 2026-08-10 (immutable historical result)
The exact 24-run batch completed on L10319 at
`data_collection/experiments/policy_corpus_v1/20260811_002551_full`. All 12,000 requested frames were written,
all 24 runs passed the online prediction/GT/result/timing gates, all CARLA actor counts returned to baseline, and
camera wait stayed healthy (run medians 24.38–35.12 ms; worst run p95 38.34 ms). One run preserved the known
post-flush CARLA 0.10 client-destructor warning; its files and actor cleanup passed before acceptance.

The immutable verification at `verification/20260811_011702/` is **FAIL_QUARANTINED** under the original gates.
This status and report were not rewritten. Its corpus-use disposition is now superseded by the corrected-goal
analysis in §9 and awaits the documented human decision:

- Pedestrian truth/matchability was achieved (11,648 eligible object-rows; 2,191 direct matches), but seven
  finite-difference samples across three runs exceeded the pre-registered 3.5 m/s all-samples ceiling. The p99
  remained about 1.70 m/s; this does not justify changing the locked gate after seeing the data.
- Vehicle held-track replay coverage was 38.79%, below the vehicle-only 45.18% legacy baseline. Pedestrian replay
  coverage was 20.75% with 4,784 observed object-frames (no legacy denominator).
- Send-needed was 0.99% versus 14.66%; selected SPLIT was 3.95% versus 5.83%. The fast-convoy send-needed rate was
  only 0.085%, and its median in-scope vehicle-frame realization was only 11.4%, so the intended persistent
  in-view urgency did not materialize.
- The shield was infeasible/over-budget on 53.08% of replay ticks while SPLIT-safe and SKIP-safe frames largely
  overlapped. The richer corpus therefore added pedestrians but did not populate the narrow state band where a
  SPLIT is safe and SKIP is unsafe.

The replacement recommendation above was made under the original send-needed framing and is retained only as
historical context. Do not execute a 24-run replacement from it. Use the §9 distributions to make a
salvage-versus-small-targeted-supplement call. The infeasible-rate observation remains relevant to eventual LOCAL
measurement, but does not authorize inventing LOCAL parameters.

## 9. Corrected-goal freshness re-score — 2026-08-11
Phase 1 is to keep localization error ≤ epsilon for **all objects in the ≤25 m/C4 scope**; phase-2 occlusion is
out of scope. The channel/forced-skip side already comes from the measured surrogate channel tables, so the CARLA
corpus is graded here for realistic motion and detection rather than engineered network/send pressure.

The table-driven re-score is at
`freshness_rescore/20260811_023817/FRESHNESS_RESCORE.md` under the immutable v1 batch. Its status is
**HUMAN_REVIEW_REQUIRED**. It uses epsilon=2 m, 20 Hz, the measured `ae32__uint4__roi0.0` base localization of
1.11 m, an instantaneous first seed, no later resend, and the locked
`sqrt(base_loc^2 + (speed * AoI)^2)` error. It reports GT-seeded motion-only and detection-seeded deployable
views separately; unmapped truth is unsafe only in the latter. Already-breached frames are separate from 3/5/10
tick near-breach frames, and breach-time estimates retain right-censored tracks.

Observed corpus distributions:

- **Speed:** pedestrians have p10/p50/p90/max 0.98/1.58/1.70/1.79 m/s; vehicles have
  0.00/~0.00/7.44/14.68 m/s. Slow (≤2 m/s) object-frames are 100% of pedestrians and 73.35% of vehicles.
  Vehicle object-frame fractions at ≥5/≥10/≥13 m/s are 16.20%/0.470%/0.086%; maximum continuous in-scope
  dwells are 4.60/1.55/0.50 s.
- **Controller-independent pressure:** GT-seeded skip-only pressure is 47.59% across all 61,372 object-frames
  (vehicles 29.21%, pedestrians 78.14%). Across the 432 seeded tracks, 359 breach and 73 are right-censored;
  Kaplan-Meier time-to-breach p10/p50/p90 is 0.25/1.00/2.05 s. Near-breach liveness at three ticks is 4.41%
  of in-scope frames, reported separately from the 72.09% of frames containing an already-breached object.
- **Deployable view:** 20,664 truth object-frames occur before detection or on never-detected tracks. Its strict
  unsafe fraction is 61.23% overall (44.39% vehicle, 89.21% pedestrian), while pressure among mapped frames is
  41.55%.
- **Detection remains a gating concern:** direct pedestrian object-row/frame coverage is 18.81%/45.60%, replay
  observation coverage is 20.75%, and 124/225 pedestrian tracks are ever detected. Vehicle values are
  34.66%/63.24%, 38.79%, and 142/207 respectively. No post-hoc pass threshold was invented.
- **Split robustness:** slow ≥0.5 s dwell appears in 10/5/4 train/validation/test runs for pedestrians and
  10/5/5 for vehicles. Sustained vehicle ≥10 m/s appears in only 2/1/1 runs; the 180 fast object-frames are
  concentrated in four fast-convoy runs (top one 46.11%, top two 71.67%). Thus validation and test each miss the
  agreed descriptive two-runs-per-split heuristic. Per-run and per-family tables are in the report/CSVs.
- **Pedestrian QC:** the same seven >3.5 m/s finite-difference samples are flagged in raw QC; all are outside
  25 m. Raw rows were preserved, and none enter the in-scope scored table.

Human call structure: (i) judge whether the observed slow-to-sustained-fast spread is usable, (ii) judge whether
47.59% GT pressure is materially above zero, and (iii) decide whether the one validation and one test fast-tail
run require a small targeted supplement. If supplementing, collect only the missing regime; do not redo all 24
runs. Pedestrian detection adequacy is a separate decision for any phase-1 pedestrian-freshness claim. This call
is now held by the §10 reconciliation; do not act on the supplement recommendation yet.

## 10. Detection reconciliation hold — 2026-08-11

The table-only analysis in `rl_agent/policy/DETECTION_RECONCILIATION.md` gives the disposition
**HOLD_DETECTION_CONFIG_RECONCILIATION**. Applying the identical direct-coverage metric to old traces proves
that offline curated recall and live actor-appearance coverage are different metrics, but does not dissolve the
problem: old vehicle cohorts score 44.70–54.95% versus 34.66% in the new corpus. An offline-visibility proxy and
timeout audit do not remove the deficit.

The earlier 0.883/0.910 reference was also the wrong model family (AE128). The exact no-AE collection checkpoint
has validated pedestrian/vehicle/overall recall of 0.855/0.893/0.879 and an identical recorded SHA-256, so this
is not evidence that the checkpoint weights changed. It is a collection-recipe regression: all new runs used
5,000 radar points/s while the validated dataset used 200,000, and live NMS-4/top-80 differs from offline
NMS-2/top-120. All 12,000 results arrived, so CPU timeouts did not create false misses. Existing tables cannot
causally separate input density, decoder capacity, and harder scenes.

Hold the pedestrian-scope call, fast-tail supplement, and controller ladder. After joint review, the smallest
admissible next experiment is a matched, pre-registered detector A/B (5k versus 200k first; decoder settings only
if still needed), not a 24-run recollection.
