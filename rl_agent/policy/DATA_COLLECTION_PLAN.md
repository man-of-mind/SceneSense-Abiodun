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

## 5. Verification gates (must pass before the corpus is used — the "are the frames sane" check)
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

## 8. Execution outcome — 2026-08-10 (candidate v1 is quarantined)
The exact 24-run batch completed on L10319 at
`data_collection/experiments/policy_corpus_v1/20260811_002551_full`. All 12,000 requested frames were written,
all 24 runs passed the online prediction/GT/result/timing gates, all CARLA actor counts returned to baseline, and
camera wait stayed healthy (run medians 24.38–35.12 ms; worst run p95 38.34 ms). One run preserved the known
post-flush CARLA 0.10 client-destructor warning; its files and actor cleanup passed before acceptance.

The immutable verification at `verification/20260811_011702/` is **FAIL_QUARANTINED**. Do not add this batch to
training/evaluation replay roots:

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

Before another expensive collection, use short smokes to (1) hold a maneuvering/accelerating target in view for
≥95% of requested frames, (2) demonstrate send-needed above baseline on the smoke replay, and (3) demonstrate
vehicle replay coverage above baseline. Separately decide prospectively whether collision-induced pedestrian
displacement belongs in the locomotion-speed gate. The 53% infeasible rate also strengthens the case for measuring
the provisional LOCAL fourth table in `REWARD_FORMULATION.md`; it does not authorize inventing LOCAL parameters.
