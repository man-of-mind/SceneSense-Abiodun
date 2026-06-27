# Cooperative Fusion — Phased Plan (DRAFT for agreement)

Goal: demonstrate that fusing per-view perception (SEG + 2D box + distance) from **two
spatially separated static sensors** yields a **more accurate object world-position** than
either view alone, and **covers occlusions** a single view cannot. Then extend to motion.

This is the SceneSense thesis in its simplest, cleanest form: infrastructure-style fixed
sensors, object-level evidence fused into a shared map.

---

## Why this should work (theory)

1. **Variance reduction.** Single-view distance error is ~1.2 m, unbiased, dominated by
   variance (we measured this). Two views with ~independent errors, averaged, cut the
   variance by ~1/sqrt(2); covariance-weighted fusion does better.
2. **Triangulation beats depth.** Each view's *bearing* (pixel -> ray via intrinsics) is
   precise; its *depth* is noisy. Two bearings to the same object **triangulate** a world
   position far better than either view's depth estimate. This is the strongest lever for
   position accuracy and the key reason cooperative fusion wins.
3. **Occlusion coverage = union of viewpoints.** An object occluded from view A but visible
   to view B still appears in the shared map. Cooperative recall = union, not min.
4. **Static ego is radar-favorable.** With a fixed sensor, moving objects have a clear
   Doppler signature against a zero-Doppler static background -> easier moving-object
   detection (the opposite of the moving-ego problem).
5. **Static removes confounds.** No per-frame ego pitch/motion blur -> clean, repeatable
   geometry. (Also lets us re-test the ground-plane depth prior, which the moving ego's
   per-frame pitch variation defeated.)

---

## Reusable assets (do not reinvent)

- `carla_collect_parked_ego_fusion_training_data.py` — static/parked-ego RGB+radar collector.
- `real_time_spatial_map_server_fusion_object_v2.py` — already aggregates multiple object
  streams (by stream-id) into one shared map. This is the multi-view aggregator.
- `carla_split_inference_udp_fusion_object_pole_client_spatial_stream{,_2,_oai}.py` — per-view
  perception clients (stream + stream_2 for two simultaneous views).
- Two parked-ego views already defined (month-2 doc): view1 spawn152 +3m right; view2
  spawn152 +8m fwd 180deg. Never run simultaneously — that is the gap we fill.
- Best perception model: archK (detection F1 0.37, SEG 0.916/0.750, 2D box, distance ~1.2 m).
- `scenesense_scenarios/scout_parked_ego_training_views.py` — to pick/validate viewpoints.

---

## Phases (gradual, each with a gate)

### Phase 0 — Calibration & sanity (PREREQUISITE; do not skip)
Two static sensors only mean something fused if both express object position in a **common
world frame**. Validate the transforms BEFORE any fusion.
- Place static ego A; place 1 car + 1 human at known CARLA positions.
- Per-view perception -> predicted object world_x/y (camera->world transform of center+distance).
- **Check: predicted world pos vs CARLA ground-truth world pos** (single view). This isolates
  transform correctness from model error. If transforms are off, fix here — fusion is
  meaningless until views agree on a static object's world position.
- GATE: single-view world-position error is explainable by the known ~1.2 m distance error
  (not a gross transform/offset bug).

### Phase 1 — Single static view + retrain + characterize
- Static ego, objects moving through the scene (parked-ego collector). Collect
  **range-balanced** data (quota per distance bin so near-field is not starved — our known
  weak spot). Static + moving objects naturally produces near AND far passes.
- Retrain fusion model (archK recipe: partial-unfreeze + distill + GIoU bbox + adaptive
  radius) on the static dataset. Eval seg / 2D box / distance **per range bin**.
- Re-test the **ground-plane depth prior** here (clean fixed camera -> it may now help).
- GATE: per-view metrics at least match the moving-ego model; near-field improved.

### Phase 2 — Two static views + fusion module (the core contribution)
- Two static egos A and B viewing the same scene from different angles (reuse the two
  predefined poses). Both stream object-level evidence (class, world pos, per-detection
  confidence) to the spatial-map server.
- **Association** (trivial for 1 car + 1 human: class + proximity). Then **fuse**, in
  increasing sophistication, measuring each:
  1. Average of world positions (baseline; validates the pipeline).
  2. **Covariance/confidence-weighted** fusion (information filter) — weight each view by
     its uncertainty (closer view / radar-supported view = more confident).
  3. **Triangulation** from the two bearings (expected best for position).
- METRIC (headline): world-position MAE vs CARLA GT for **view A alone / view B alone /
  fused**. Success = fused < min(A, B). Report per fusion method.
- GATE: fused position error < single-view; demonstrate the cooperative gain quantitatively.

### Phase 3 — Occlusion demonstration (thesis headline)
- Place an occluder so the object is hidden from view A but visible to view B.
- Show the shared map still contains the object (cooperative coverage), and the fused
  position is correct from B alone, refined when A re-acquires.
- This is the "see around the occlusion" result the whole project is about.

### Phase 4 — Gradual motion
- Add object motion (map carries velocity; simple tracking/association across frames).
- Then add ego motion (re-introduce the moving-ego confounds we already characterized).

---

## Fusion module (new code, the deliverable)
- Input: per-view detections (world_x/y, class, score, optional per-axis uncertainty, bearing).
- Association: class-aware nearest-neighbor (Hungarian later for many objects).
- Estimators: mean -> covariance-weighted (information filter) -> bearing triangulation.
- Output: fused object (world pos + fused covariance) into the shared map.
- Evaluate against CARLA GT; ablate fusion method and number of views.

## Risks / honest caveats
- Transform/calibration errors masquerade as "fusion doesn't help" -> Phase 0 guards this.
- Association is trivial now (2 objects) but is a real problem at scale -> keep scenes simple first.
- Distance is variance-limited single-view; triangulation is the lever -> if averaging barely
  helps, triangulation is the theoretically-correct next step, not more model training.
- Static-first means results may not transfer directly to moving ego -> Phase 4 re-validates.

## Success criteria
- Phase 0: views agree with GT on a static object's world position (transforms correct).
- Phase 2: fused world-position MAE < single-view (quantified, per method).
- Phase 3: object present in shared map while occluded from one view.

---

## Agreed decisions (2026-06-26)
- Start: **Phase 0 calibration first**.
- Primary fusion deliverable: **triangulation from bearings** (build average -> covariance ->
  triangulation, but triangulation is the target).
- Execution: **step-by-step with review** — run one phase/sub-step, report, then proceed.

### Phase 0 sub-steps
0a. Transform sanity from EXISTING eval data (no CARLA): check the world-coordinate bias
    (mean pred_world - gt_world) on the archK eval CSV. A gross transform bug shows as a
    systematic x/y offset or rotation. ~0 bias => camera->world transform is sound.
0b. Controlled static scene (CARLA): static ego + 1 car + 1 human at known poses; confirm
    predicted world pos vs CARLA GT per view, within the expected ~1.2 m model error.
