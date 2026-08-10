# Data-collection plan — richer CARLA corpus to build the environment cleanly (DRAFT for codex review)

**Why:** the current replay corpus (staleness study) is the binding limit on the policy environment. Inspection
(2026-08-10) confirmed the traces are physically sound BUT: (a) **ground truth is vehicles only — no pedestrian
GT**, so pedestrian localization/recall cannot be scored at all; (b) sends are sparse (~5% SPLIT) so safety
denominators are thin; (c) observation coverage ~45%. We collect a purpose-built corpus that fixes (a)–(c) so the
environment covers the safety-critical cases it is currently blind to. Real CARLA data is the primary path
(internship extended +3 months → do it properly, not a synthetic stub).

**Scope discipline:** this is a *data* task, not a new pipeline. Reuse the existing staleness collector; the only
real delta is spawning + logging pedestrians and choosing richer scenarios. Do NOT expand beyond the scenarios
below without a note here.

## 1. Reuse, don't rebuild
- Base collector: `uplink_only_spatial_map_pipeline/carla_fusion_staleness_scenario_uplink_only.py` (produces the
  exact `*_object_ground_truth.csv` + `*_object_predictions.csv` schema the surrogate already parses).
- Pedestrian-spawn reference (already working in-repo): `radar_camera_lidar_data_collect_update_pedestrian_
  vizualizor_fusion.py` and `carla_collect_moving_ego_fusion_training_data.py`.
- Per repo convention: **copy** the collector into `abiodun/data_collection/` and edit the copy — never edit the
  top-level/shared originals.

## 2. The only functional change: log pedestrian ground truth
The detector already emits `person` predictions (confirmed in the current prediction CSVs), so perception needs
no change. The gap is purely on the GT side:
- Spawn a controllable number of **walkers** (pedestrians) with the CARLA walker AI, near roads/crossings within
  the ego's field of view and ≤ 25 m.
- Log pedestrian actors into `*_object_ground_truth.csv` with the **same columns and the actor-origin position
  convention** already used for vehicles (`origin_x/origin_y`, `class_name=pedestrian`, `distance_m`,
  `in_camera_frustum`, size fields). Do not switch to bbox-center (reintroduces the known 1–1.3 m bias).
- Everything else (frame cadence ~8–10 fps, ego autopilot, vehicle spawns, streams layout) stays as in the base
  collector.

## 3. Scenarios (target ~20–30 runs, ~50 s each — matches existing trace length)
Three families, split into train/val/test by **family**, never by frame:
1. **`ped_crossing`** — pedestrians (5–15) crossing/walking near the ego path, moderate vehicle traffic. The
   critical missing class. Ensures a real pedestrian-recall + pedestrian-localization denominator.
2. **`dense_fast`** — denser traffic (raise vehicle spawn count) and faster ego + actors (target 30–50 mph where
   the map legally allows). Forces frequent sends → fixes the thin SPLIT denominator and exercises
   localization-under-latency where ε actually binds.
3. **`mixed_urban`** — balanced veh + ped at nominal density/speed, the "typical operating point" for headline
   numbers.
Aim for meaningfully higher **observation coverage** than the current 45% by keeping actors in-FOV and ≤ 25 m;
report the achieved coverage per run.

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
- **Pin the perception/back-half to a separate GPU** (or throttle nothing on the render GPU) so CARLA render
  contention cannot silently drop the frame rate — the bug that corrupted an earlier grid (front dropped to
  ~2.5 fps). Watch `camera_frame_wait` ≈ 32 ms, not ~122 ms.
- Check `/proc/loadavg` + `docker ps` first; **reuse** a running CARLA server; never kill another user's
  CARLA/OAI. Town10HD_Opt for consistency with prior work.
- This is uplink/collection only — no OAI closed-loop needed for the corpus itself.

## 5. Verification gates (must pass before the corpus is used — the "are the frames sane" check)
Emit a `CORPUS_VERIFICATION.md` per collection batch with:
1. **Pedestrian GT present:** every `ped_crossing`/`mixed_urban` run has `class_name=pedestrian` rows in GT with
   plausible speeds (0.5–2.5 m/s) and sizes (height ~1.6–1.9 m, width ~0.4–0.7 m).
2. **Position sanity:** `origin_x/origin_y` within town bounds; `distance_m` ≥ 0; no NaN explosions in GT
   positions; `in_camera_frustum` ∈ {0,1}.
3. **GT↔prediction matchability:** run the existing staleness matcher (actor-origin, score ≥ 0.20, 5 m gate) and
   report per-class **observation coverage** — pedestrians must actually match, not just exist in GT.
4. **Send-path exercised:** on a quick surrogate pass, the fraction of frames where a send is *needed* (ε binds)
   is materially above the current corpus, i.e. SPLIT is no longer ~5%.
5. **Speed/density achieved:** report per-family actor counts + speed distributions vs the targets in §3.

## 6. After collection
1. Point the surrogate `replay.roots` at the new corpus (keep the old vehicle-only set as a labelled legacy
   comparison).
2. Re-run the ε2/core90/range25 pilot on the new corpus → confirm pedestrian metrics now have real denominators
   and the shield's safety numbers are no longer denominator-starved.
3. THEN proceed to the controller ladder (rule → bandit → MPC → RL) on the clean environment.

## 7. Open questions for Abiodun / advisor
- Pedestrian-recall **hard floor** value (safety-critical class) — still advisor-pending; this corpus is what
  makes it measurable.
- Confirm 25 m as the headline range (data already favors it; 40 m stays diagnostic).
- Who runs CARLA: **decided — codex runs the full loop on L10319** (its CARLA 0.10.0), so collection and
  consumption share one box/version. local Claude reviews the plan + verification output.
