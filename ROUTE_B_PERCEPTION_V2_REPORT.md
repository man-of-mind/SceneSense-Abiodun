# Route B perception dataset v2 — implementation and smoke report

**Date:** 2026-08-24 · **Scope:** Phase 1 radar-input correction, Phase 2 collection smokes,
Phase 3 training preparation. No production edits, no checkpoint overwrite, no locked-test
evaluation, no OAI run, no campaign execution.

---

## 0. What changed and where

| Artifact | Status | Purpose |
|---|---|---|
| `data_collection/radar_sweep_aggregator_v1.py` | new | Timestamp-binned 100 ms logical sweeps, motion-compensated 200 ms temporal window, 20/10/5 Hz cadence bookkeeping |
| `data_collection/render_provenance_v1.py` | new | Epic-quality preflight gate + manifest provenance + RGB/segmentation frame validation |
| `data_collection/run_route_b_radar_cadence_smoke_v1.py` | new | Phase-1 standalone radar smoke |
| `data_collection/run_route_b_perception_collection_v2.py` | new (v1 retained untouched) | Canonical episode collector on the v2 contract |
| `data_collection/configs/route_b_perception_v2.yaml` | new | Resolved collection config |
| `pole_lraspp_multimodal_fusion/object_head_pilot_v1/` | new | Phase-3 pilot: target variant, runner, trial configs, GPU probe |

Unchanged and deliberately not touched: `run_route_b_density_loop.py` (SHA-256
`f2abd86c…5730`, re-verified), `generate_traffic_v1` import, `TrafficPopulationManager`,
`PopulationLedger`, `object_targets.py`, `train_fusion.py`, `radar_fusion.py`, and every
existing checkpoint.

### Traffic helper (confirmation only)

`data_collection/generate_traffic_v11.py` was diffed against
`rl_agent/advisor_helper_scripts/codes/generate_traffic_v1.py`: identical apart from one
trailing blank line (`811d810 < `). The density runner's `import generate_traffic_v1` is
unchanged and the helper is not duplicated. `--no-hybrid-physics` and population
reconciliation are retained together.

---

## 1. Phase 1 — radar-input correction

### 1.1 The contract as implemented

* World tick 20 Hz (`fixed_delta_seconds = 0.05`), synchronous.
* Radar `points_per_second = 200,000` — **never inflated**. Verified by reading the
  attribute back off the spawned sensor and gating on it (`pps_not_inflated`).
* Radar `sensor_tick = 0.0` (free-running) → one raw callback per world tick.
* A logical sweep is the half-open 100 ms bin `(anchor + 0.1·(i−1), anchor + 0.1·i]`, where
  `anchor` is the timestamp of the first prepared-input tick, so a bin always closes exactly
  on a tick where a model input is due. **Every callback whose timestamp falls in the bin is
  accumulated**; the count per bin is measured and gated, never assumed from `sensor_tick`.
* Prepared model input at 10 Hz from the current **and** immediately previous sweep →
  contiguous 200 ms of support.
* Persistence at 5 Hz (every second prepared input).
* Radar tensor shape and channels unchanged: `4 × 432 × 768` float32, same
  occupancy / inverse-range / radial-velocity / stationary-age semantics, produced by the
  unmodified `build_radar_sample`.

### 1.2 Motion compensation

Returns from the previous sweep were captured from a different ego pose. Each raw callback
is lifted to world coordinates using **its own** callback transform at ingest, then the
accumulated window is re-expressed in the radar pose at the prepared tick and converted back
to CARLA's `[altitude, azimuth, depth, velocity]` layout. This means the existing
`build_radar_sample` is called unchanged. Measured round-trip error: **3.6 × 10⁻⁵ m**.

### 1.3 Two defects found and fixed

**(a) `sensor_tick = 0.1` cameras skip captures.** Measured directly over 200 world ticks:
gap histogram `{2: 98, 3: 1}`. One skipped capture permanently shifts the 10 Hz phase, after
which an exact-frame fetch fails on every subsequent prepared tick. Free-running sensors
measured `{1: 199}` — exact. Cameras are therefore free-running and the 10 Hz selection is
derived from world ticks, the same rule already required for radar. Cost: the three cameras
render at 20 Hz instead of 10 Hz.

> This also affects the **existing v1 collector**, which still uses `sensor_tick = 0.1`
> together with an exact-frame assertion. It is a latent abort/misalignment risk there.

**(b) Sweep-bin drift.** CARLA reports elapsed time as a float32-derived double
(0.05000000074505806 s per tick), so the error against the sweep anchor accumulates without
bound (~7 × 10⁻⁷ s over 900 ticks). An initial 1 × 10⁻⁶ s bin tolerance flipped a boundary
partway through a 45 s run, splitting sweeps into one callback each and shrinking windows
from 4 callbacks to ~3.2. The tolerance is now **a quarter of a world tick** (0.0125 s) —
far larger than any drift, far smaller than the half-tick spacing between a boundary
callback and an interior one.

### 1.4 Result — `RADAR_SMOKE_PASSED` (19/19 gates)

`data_collection/route_b_perception_v2/radar_cadence_smoke_v1.json`, 46 s simulated,
Town10HD_Opt, moving ego at 25 km/h.

| Measured quantity | Result |
|---|---|
| Configured PPS | **200,000** (read back from the sensor) |
| World tick rate | **20.0 Hz** — 920 ticks, interval 0.05000000074 s (min = mean = max) |
| Raw callback cadence | **20.0 Hz** — 920 callbacks, one per world tick |
| Returns per raw callback | mean **9,180** (min 8,415 / max 9,964) |
| Logical sweep cadence | **10.0 Hz** — 451 sweeps |
| Callbacks per 100 ms logical sweep | **exactly 2** on every consumed sweep |
| Returns per logical sweep | mean **18,340** (min 16,942 / max 19,925) |
| Temporal-window timestamp span | **0.150 s** first→last callback (min = mean = max); 200 ms support; **4** callbacks; mean **36,678** returns |
| Prepared-input cadence | **10.0 Hz** — 450 |
| Saved-frame cadence | **5.0 Hz** — 225 |
| Dropped / duplicate / out-of-order callbacks | **0 / 0 / 0** |
| Timestamp reversals | **0** (callback, tick, prepared and saved streams) |
| Sensor-frame alignment | all four sensors on the same world frame at every prepared tick |
| Timestamp alignment | camera↔radar Δt = **0.0 s**; world-snapshot Δt = **0.0 s** |

Returns per callback are ~9,180 rather than a nominal 10,000 because CARLA discards returns
that hit no geometry; the value is scene-dependent and is reported, not assumed.

**Note on returns per sweep.** The full trailing bin left over after the last prepared input
is reported but not gated — no prepared input ever consumes it, so requiring it to be
complete would fail a correct run.

---

## 2. Phase 2 — collection smokes

### 2.1 Hard gates implemented per episode

Written to `route_summary.json` as an explicit `gates` object with an overall
`COLLECTION_EPISODE_PASSED` / `FAILED` status, and reflected in the process exit code:

route/map/hash and waypoint coverage · exact requested population at episode start ·
population IDs, losses and replenishments accounted for (`PopulationLedger`, retained) ·
≥95 % of each requested population alive at **every saved frame** · no population deficit
longer than `replenish_interval + 2 s` · no watchdog abort or route incompletion · zero
missing or corrupt saved records · exact sensor-frame alignment and bounded timestamps ·
the full Phase-1 cadence contract re-gated in-episode · all sensors, ego, NPCs and walkers
cleaned up · CARLA/world settings restored.

### 2.2 Collisions

A logged collision alone is **not** a hard failure when the route completes and the data
remain valid. Collision timestamps are preserved and a per-sample incident flag is written
to `collision_incident_windows.csv` (`sample_id, frame_id, timestamp_s, incident_window,
nearest_collision_dt_s`) with a ±2 s half-width. **No frame is deleted.** Primary metrics
are to be reported on all frames, with a separate sensitivity result excluding the ±2 s
windows. A collision that causes a stall, route incompletion or corrupted alignment still
fails via `route_completed` / `no_watchdog_abort` / `sensor_frames_exactly_aligned`.

### 2.3 Epic rendering provenance (mandatory)

`render_provenance_v1.py` confirms the renderer **from the controlled launch
configuration** — the command line of the `CarlaUnreal-Linux-Shipping` process bound to the
RPC port in use — because quality level is a server launch flag and is not exposed on
`carla.WorldSettings`. Collection aborts unless:

* `-quality-level` is present **and explicitly** `Epic` (a missing flag is a failure, not a
  default);
* no `-nullrhi` / `-norender` / `-disablerendering` / `-noRHI`;
* `world.get_settings().no_rendering_mode` is `false`.

`-RenderOffScreen` is permitted and recorded separately — it is headless GPU rendering, not
no-rendering mode. The full launch command, quality level, rendering mode, resolution, FOV,
weather, map, server and client version are written verbatim into every episode's
`metadata.json` and `route_summary.json`. Camera resolution, FOV, blueprint settings and
weather are fixed by the resolved config and identical across episodes.

RGB and segmentation frames are validated at **every prepared input** (10 Hz): correct size,
correct payload length, non-empty, same frame index, same timestamp.

### 2.4 Storage and throughput (measured)

From the 61 s plumbing integration check (306 saved frames, all cadence/population/alignment
gates green; only `route_completed` failed, as designed for a truncated loop):

| Component | Per saved frame |
|---|---|
| `radar_tensors/*.npy` (4×432×768 float32, uncompressed) | **5.1 MB** |
| `radar_points/*.npz` (compressed, ~36.7 k points) | **1.3 MB** |
| `rgb/*.jpg` (1280×720, q92) | **324 KB** |
| `semantic_tags/*.png` | 46 KB |
| `masks/*.png` | 4 KB |
| **Total** | **≈ 7.03 MB** (min 6.78 / max 7.16) |

At 5 Hz over a Route B loop the per-episode figures are in §2.5 (measured on the real
episodes). The **radar tensor npy dominates at 72 % of the corpus** — it is a sparse
4-channel raster stored uncompressed. Switching that one file to `np.savez_compressed`
would cut episode size several-fold losslessly, but it changes the loader's read path, so
it is raised as a decision rather than applied.

The depth camera is spawned and rendered but its frames are **discarded** (v1 did the same).
It is the one signal already being paid for that would allow a true visibility/occlusion
label to be derived. Not saved here — that would be inventing a schema field mid-collection.

### 2.5 Episode results — `COLLECTION_SMOKE_FAILED`

**traffic_30_30, split `smoke`, seeds 101/1101** —
`data_collection/experiments/route_b_perception_v2/20260824_smoke_traffic_30_30/`
→ `COLLECTION_EPISODE_FAILED`, process exit 1.

**18 of 20 gates passed.** The route itself was clean: completed, 19/19 waypoints,
1251.68 m driven, 362.5 s simulated, return position error 0.554 m, no watchdog abort, no
roadblock intervention, sensors/actors cleaned up, settings restored.

The v2 timing contract held exactly over the full episode:

| Quantity | Result |
|---|---|
| World ticks | 7,250 @ **20.0 Hz** |
| Raw radar callbacks | 7,270, mean **9,308** returns (min 7,253 / max 10,000) |
| Logical sweeps | 3,626 @ **10.0 Hz**, **exactly 2 callbacks each**, mean **18,618** returns |
| Prepared inputs | 3,625 @ **10.0 Hz**, **exactly 4 callbacks**, span **0.150 s**, mean **37,237** returns |
| Saved frames | **1,813 @ 5.0 Hz** |
| Dropped / duplicate / out-of-order / reversed | **0 / 0 / 0 / 0** |
| Sensor frame + timestamp alignment | max Δt **0.0 s**; 3,625/3,625 RGB+segmentation frame checks passed |
| Storage | **11.78 GiB**, mean **6.98 MB**/frame |
| Wall clock | 1,961 s driving (RTF ≈ 0.185); prepared-input assembly mean **0.391 s** |

No I/O backpressure and no dropped frames were observed, so an asynchronous data writer is
**not** justified.

19 collision contacts were logged across two events (a taxi at t≈71 s, a police charger at
t≈356 s). Per the collision policy these are **not** a failure: the route completed and the
data stayed valid. 45 of 1,813 frames fall in a ±2 s incident window and are flagged, not
deleted.

#### The two failed gates

```
population_alive_95pct_every_saved_frame          False
no_population_deficit_beyond_replenish_plus_2s    False
```

Pedestrians decayed monotonically and were **never replenished**:

| t (sim s) | 0 | 30 | 60 | 120 | 180 | 240 | 300 | 362 |
|---|---|---|---|---|---|---|---|---|
| pedestrians alive | 30 | 28 | 28 | 27 | 24 | 23 | 22 | **21** |
| vehicles alive | 30 | 30 | 30 | 30 | 30 | 30 | 30 | 30 |

Mean 24.9/30 (83 %), minimum 21/30 (70 %), 1,707 of 1,813 saved frames below the 29-actor
floor, one continuous deficit span of **341.4 s** against a limit of `replenish_interval + 2`
= 7 s. Vehicles were exactly 30 on all 1,813 frames.

#### Root cause — `world.get_actors(id_list)` returns stale `is_alive`

The `PopulationLedger` reported **0 pedestrians lost, 0 replenished, `live_min` 30** for the
same episode. Ledger and world truth cannot both be right, so the two were measured against
each other directly
(`data_collection/route_b_perception_v2/walker_presence_diagnostic_v1.py`, no ego, no
collisions, 120 s):

| Count of the same 30 owned walker bodies | at t = 119 s |
|---|---|
| A — `world.get_actors().filter("walker.pedestrian.*")` | **27** |
| B — `world.get_actors(owned_ids)` filtered by `is_alive` | **30** |
| C — per-id `world.get_actor(id).is_alive` | **27** |
| walkers alive per C but absent from A | **0** |

A and C agree; B does not. `world.get_actors(<explicit id list>)` hands back actor objects
whose `is_alive` is **stale**, so three dead walkers still report alive. Cleanup then
printed `3 managed walkers could not be destroyed` — exactly those three phantoms.

`TrafficPopulationManager._live_actor_map` is built on precisely that call, so
`_reconcile_owned_actors` never detects a dead walker, never drops it from `self.walkers`,
and therefore `missing_walkers = target - len(self.walkers)` stays at 0 and `_spawn_walkers`
is never called. `PopulationLedger` observes the manager's ownership set, so it inherits the
same blindness and reports a healthy population that does not exist.

This is a **pre-existing defect in the shared traffic helper**, not in the v2 collector or
the density runner. Vehicles mask it because no vehicle died in this episode (0 lost). Walker
attrition is intrinsic — the diagnostic lost 3 walkers in 120 s with no ego present at all.

#### Smallest fix

Feeding the existing reconciler a correct liveness signal, without editing
`generate_traffic_v1.py`, duplicating it, or changing the ledger. Wrap `population.reconcile`
in the v2 collector so the prune happens **inside** the ledger's before/after snapshot
window, which keeps loss/replenishment accounting correct:

```python
def _install_walker_liveness_fix(population, world):
    """Drop walker records whose body is absent from the world snapshot.

    world.get_actors(<id list>) returns stale is_alive, so the manager's own
    reconciliation never sees a dead walker and never replenishes it. Pruning
    here lets the UNMODIFIED reconcile()/_spawn_walkers() path do the refill.
    """
    original_reconcile = population.reconcile

    def reconcile_with_liveness_prune():
        present = {int(a.id) for a in world.get_actors().filter("walker.pedestrian.*")}
        population.walkers = [
            record for record in population.walkers
            if record.get("id") is None or int(record["id"]) in present
        ]
        return original_reconcile()

    population.reconcile = reconcile_with_liveness_prune
```

`population` is not currently handed to the collector, so this needs the density runner's
`maintain_population` closure or the manager instance to be reachable — roughly 15 lines in
`run_route_b_perception_collection_v2.py` plus one hook.

**Not applied.** It changes scenario population behaviour (pedestrians would now be
continuously respawned mid-episode, which alters pedestrian density statistics and
introduces actors that appear from nothing near the ego), and the instruction was to retain
the existing reconciliation untouched. That is a scope decision, so the alternatives are put
below rather than chosen.

#### Decision required before the campaign

1. **Apply the liveness fix** — holds true 30/30 and 50/50 density, at the cost of
   mid-episode pedestrian respawns.
2. **Fix `generate_traffic_v1.py` itself** (one line in `_live_actor_map`: build the map from
   the world snapshot instead of trusting `is_alive`). Correct at the source and fixes the
   ledger too, but it is a shared production file behind accepted artifacts.
3. **Relax the gate** — accept ~70–83 % pedestrian retention as the scenario's real
   behaviour. Cheapest, but pedestrian recall is a headline metric for this dataset and the
   pedestrian population would be decaying through every episode.

Recommendation: **option 2**, with option 1 as the no-production-edit fallback. Either way
the fix must land before collection, because the defect biases the pedestrian half of the
corpus.

**traffic_50_50: not run.** The instruction is to stop on the first failure.

---

## 3. Phase 3 — training preparation (nothing trained)

### 3.1 Warm start and the frozen baseline

* Warm start: `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`
  — SHA-256 `f319e2a5e8fb134e74c24c0822233e17368df6e4c733add658026603e131d4fa`, verified.
  Used as `init_rgb_checkpoint` **and** `init_object_checkpoint`. It is never written to;
  the pilot runner refuses to start if the target trial's `best.pt` already exists.
* noAE is used for the first architecture pilot because it isolates the object-training
  change from AE compression (`ae_bottleneck: 0`).
* The recipe is the recorded noAE recipe (`rl_agent/ae_integrated/mprime_joint_noae.json`):
  AdamW, lr 1.5e-4, wd 1e-4, strong photometric augment, no geometric augment,
  768×432, cosine schedule, `freeze_bn: true`, `predict_bbox2d: true`,
  `adaptive_heatmap_radius: true`, `max_gt_distance_m: 40`.

### 3.2 Architecture pilot — the two arms

| | Arm A (control) | Arm B (candidate) |
|---|---|---|
| Config | `configs/pilot_A_control_objhead_smoke_v1.json` | `configs/pilot_B_capped_objhead_smoke_v1.json` |
| Heatmap radius | current adaptive behaviour | adaptive, with `vehicle_heatmap_radius_cap_px = 4` |
| Person radii | adaptive | adaptive (**unchanged**) |
| Code path | unmodified production `build_object_targets` | pilot `build_object_targets_capped` |
| Everything else | identical, identical seed | identical, identical seed |

The candidate lives in `pole_lraspp_multimodal_fusion/object_head_pilot_v1/target_variants_v1.py`,
**outside** the production package, so no production file is edited. Verified on a
deterministic synthetic frame:

```
PARITY OK: cap disabled is bit-identical to production
  center_heatmap     identical=False      <- the only difference
  regression         identical=True
  regression_mask    identical=True
  gt_objects         identical=True
  gt_class_indices   identical=True
  gt_count           identical=True
vehicle heatmap >0 px: control 3609 -> capped  972
person  heatmap >0 px: control 2914 -> capped 2914   <- person targets untouched
```

Tensor shapes, the decoder, the regression heads and all person targets are unchanged. The
runner asserts this parity **before any training step runs**, so a control/candidate
difference can never be a copy bug.

> Alternative worth deciding before training: a three-line, default-off keyword argument in
> `object_targets.py` would remove the duplicated function entirely. It is a production edit,
> which is why it was not taken.

**Pilot protocol.** Short, identical-seed, object-head-only smoke for both arms on the train
episodes; evaluate on **complete validation episodes**. Arms run **sequentially** on the GPU.
**If the cap gives no directional vehicle-precision / duplicate-FP improvement, stop before
full training.**

### 3.3 Selected curriculum

1. **Object-head stage** (`configs/curriculum_stage1_objhead_v1.json`) — backbone and
   segmentation classifier frozen (`freeze_backbone: true`, `freeze_classifier: true`),
   object head trained with the selected target construction. lr 1.5e-4, 40 epochs.
2. **Joint-refinement stage** (`configs/curriculum_stage2_joint_v1.json`) — starts from the
   stage-1 checkpoint; backbone, segmentation classifier and object head all unfrozen;
   **batch-norm state stays frozen** (`freeze_bn: true`); **lower learning rate** (3e-5 vs
   1.5e-4); segmentation retained as a **secondary** loss (weight 0.3 vs 1.0). 25 epochs.

The object-stage checkpoint is compared against the joint-refined checkpoint on the **same
complete validation episodes**; that comparison decides whether strict staged or
staged-plus-joint training is retained. Both checkpoints are kept.

### 3.4 Metrics and promotion rules

**Primary:** vehicle/person XY MAE · vehicle/person precision, recall, F1 · FP/frame ·
duplicate-FP fraction.
**Guarded secondary:** 2D centroid error · dimension MAE · vehicle/person IoU and mIoU ·
inference and decoder latency.

**Promotion (all must hold, vs the frozen old noAE baseline on the same validation episodes):**

| Rule | Threshold |
|---|---|
| Vehicle precision | must **improve** |
| Duplicate-FP fraction | must **fall** |
| Vehicle / person recall | may not fall > **0.02** absolute |
| Vehicle XY MAE | may not worsen > **0.05 m** |
| Person XY MAE | may not worsen > **0.10 m** |
| Vehicle / person IoU | may not fall > **0.02** absolute |
| Dimension MAE | may not worsen > **10 %** |

Reported **separately**, as aspirational floors and not as gates: vehicle recall 0.90,
person recall 0.85, vehicle XY MAE 0.90 m, person XY MAE 1.20 m.

**The locked test split (bundle 3) is not evaluated or inspected until explicitly
authorized.**

### 3.5 GPU plan on the 24 GB card

* AMP on (`training.amp: true`, `GradScaler` + `autocast` already in `train_fusion.py`).
* **One GPU training process at a time**; arms never run concurrently.
* Batch-size probe at 16 / 24 / 32 via
  `object_head_pilot_v1/probe_gpu_batch_v1.py`; selection rule is the largest stable batch
  still leaving ≥ 15 % of the card free. Records peak allocated **and** reserved memory,
  samples/s and GPU utilization per batch size.
* DataLoader starts at `num_workers: 8`, `prefetch_factor: 2`, `persistent_workers: true`,
  `pin_memory: true` — and the probe **measures** worker counts against the real dataset
  rather than assuming more is faster (`--dataset-dir --worker-counts 4 8 12`).
* Environment recorded per run by `run_object_head_pilot_v1.py`: Python, PyTorch, CUDA,
  cuDNN, GPU name/capability/memory, driver, TF32 matmul/cuDNN flags,
  `float32_matmul_precision`, cuDNN benchmark/deterministic, `CUDA_VISIBLE_DEVICES`,
  `CUBLAS_WORKSPACE_CONFIG`.
* An asynchronous data writer is **not** added — it is only justified if the collection
  smoke shows I/O backpressure or dropped frames (see §2.5).

This machine: Python 3.10.x, PyTorch 2.11.0.dev20260120+cu128, CUDA 12.8, cuDNN 9.15.01,
NVIDIA GeForce RTX 5090 Laptop GPU, 24463 MiB. The probe was **not** run during collection —
CARLA is rendering at Epic on the same GPU and would corrupt both measurements.

---

## 4. Dataset splits

* Splits are by **complete episode**, never by frame.
* Registered bundles (`SEED_BUNDLES` in the collector, mirrored in the resolved config):

| Bundle | scenario seed | TM seed | Split |
|---|---|---|---|
| 1 | 101 | 1101 | train |
| 2 | 202 | 1202 | val |
| 3 | 303 | 1303 | **locked test** |
| 4 | 404 | 1404 | train (only if the architecture pilot passes) |

* Initial pilot corpus: bundles 1–3 × {`traffic_30_30`, `traffic_50_50`} = **6 episodes**.
* Final corpus: + bundle 4 × both densities = **8 episodes**.
* Collection is **never** multiplied by AE, quantization, ROI or network profile — every
  model/profile evaluation reuses the same collected frames.
* Episodes collected with `--split smoke` (no bundle) are labelled `smoke` in
  `metadata.json` and are not admissible as a canonical split.

## 5. Ground truth

* Convention: **actor origin**. Raw GT preserved at collection; `gt_max_distance_m = 140`.
* Collection is **not** restricted to actors in the driving lane — that would bias the
  detector. The eligibility filter is applied at **evaluation** time only: camera-frustum /
  projected-centre, projected area ≥ 12 px, distance ≤ 40 m.
* Person box-mask semantics preserved and their provenance recorded explicitly in
  `metadata.json`: semantic-tag training mask, then person regions overpainted as filled
  axis-aligned boxes from projected actor bounding boxes
  (`rasterize_person_regions(shape="box")`).
* **Visibility/occlusion:** the historical object-box schema (45 columns) carries no explicit
  visibility or occlusion label. The available related evidence — `gt_bbox_area_px`,
  `gt_distance_m`, `radar_support_points`, `stationary_age_s` — is recorded. No occlusion
  label is invented. Deriving one would require persisting the depth frames, which is
  flagged in §2.4 as a decision, not taken.

---

## 6. Exact commands

### 6.1 CARLA server (mandatory for every collection and validation run)

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
setsid ./CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000 \
  >/tmp/carla_route_b_perception_v2.log 2>&1 &
```

`-quality-level=Epic` is **required and explicit**; `-quality-level=Low` is never used.
`-RenderOffScreen` keeps GPU rendering on and is not no-rendering mode. A fresh CARLA
process is started for **every** episode; the collector reloads Town10HD_Opt itself.

### 6.2 Phase 1 radar smoke

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
V=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3
env -u PYTHONPATH $V data_collection/run_route_b_radar_cadence_smoke_v1.py --seconds 45
```

Do **not** export `PYTHONPATH` for any CARLA client here — it shadows `abiodun/` with the
stale `neu_collab/` copy.

### 6.3 Preflight only (renderer + route hashes, no world load)

```bash
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_30_30 --output-dir /tmp/probe --preflight-only
```

### 6.4 Canonical collection — one fresh CARLA process per episode

Run these **one at a time**, restarting CARLA (§6.1) before each.

```bash
V=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3
ROOT=data_collection/experiments/route_b_perception_v2

# --- bundle 1 = TRAIN ---
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_30_30 --seed-bundle 1 --output-dir $ROOT/20260825_b1_traffic_30_30_train
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_50_50 --seed-bundle 1 --output-dir $ROOT/20260825_b1_traffic_50_50_train

# --- bundle 2 = VALIDATION ---
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_30_30 --seed-bundle 2 --output-dir $ROOT/20260825_b2_traffic_30_30_val
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_50_50 --seed-bundle 2 --output-dir $ROOT/20260825_b2_traffic_50_50_val

# --- bundle 3 = LOCKED TEST (collect, then do not open) ---
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_30_30 --seed-bundle 3 --output-dir $ROOT/20260825_b3_traffic_30_30_test
env -u PYTHONPATH $V data_collection/run_route_b_perception_collection_v2.py \
  --density traffic_50_50 --seed-bundle 3 --output-dir $ROOT/20260825_b3_traffic_50_50_test
```

Bundle 4 (`--seed-bundle 4`, both densities) is collected **only if the architecture pilot
passes**. `--target-speed-kph` defaults to 25.0 and `--no-hybrid-physics` is the default;
neither needs to be passed. Every episode exits non-zero unless
`route_summary.json → status == COLLECTION_EPISODE_PASSED`.

Per-episode acceptance check:

```bash
$V -c "import json,sys;s=json.load(open(sys.argv[1]));\
print(s['status']);print([k for k,v in s['gates'].items() if not v])" \
  $ROOT/<episode>/route_summary.json
```

### 6.5 Phase 3 — after the dataset is copied and approved

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
export PYTHONPATH="$PWD/pole_lraspp_multimodal_fusion:$PWD:$PWD/rl_agent/feature_ae"
export CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/matplotlib-cache QT_QPA_PLATFORM=offscreen
V=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python3
P=pole_lraspp_multimodal_fusion/object_head_pilot_v1

# 1. batch-size probe (no CARLA running on the GPU)
env -u PYTHONPATH $V $P/probe_gpu_batch_v1.py --batch-sizes 16 24 32
# ...then, once the dataset is in place, measure the loader too:
env -u PYTHONPATH $V $P/probe_gpu_batch_v1.py --batch-sizes 16 24 32 \
  --dataset-dir <merged-dataset> --worker-counts 4 8 12

# 2. architecture pilot, SEQUENTIAL - arm A then arm B
$V $P/run_object_head_pilot_v1.py --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --trial-json $P/configs/pilot_A_control_objhead_smoke_v1.json \
  --experiment-dir experiments/object_head_pilot_v1/arm_a_control --training-budget-hours 1.0
$V $P/run_object_head_pilot_v1.py --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --trial-json $P/configs/pilot_B_capped_objhead_smoke_v1.json \
  --experiment-dir experiments/object_head_pilot_v1/arm_b_capped --training-budget-hours 1.0

# 3. curriculum (only if the pilot shows directional improvement)
$V $P/run_object_head_pilot_v1.py --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --trial-json $P/configs/curriculum_stage1_objhead_v1.json \
  --experiment-dir experiments/object_head_pilot_v1/stage1_objhead
# set curriculum_stage2_joint_v1.json init_*_checkpoint to stage1's best.pt, then:
$V $P/run_object_head_pilot_v1.py --config pole_lraspp_multimodal_fusion/configs/fusion_full_run.yaml \
  --trial-json $P/configs/curriculum_stage2_joint_v1.json \
  --experiment-dir experiments/object_head_pilot_v1/stage2_joint
```

`--dry-run` resolves the trial, runs the target-parity guard and records the environment
without training a step.

---

## 7. Consequences of the v2 contract that a consumer must know

1. **Route B v1 and Route B v2 corpora must not be mixed.** The Route B v2 radar tensor
   integrates 200 ms of support instead of the single ~100 ms callback used by the Route B
   v1 collector, so `radar_points` per sample roughly doubles (Route B v1 ≈ 18–20 k, v2
   ≈ 33–40 k). The same accumulated cloud is passed to `build_object_rows`, so the GT
   annotation `radar_support_points` — and the `radar_support` regression target derived
   from it — is systematically larger than in a Route B v1 episode. Within v2 the signal is
   consistent; across Route B v1/v2 it is not.

   This is **not** a claim of incompatibility with every historical corpus. The historical
   M-prime training corpus already used a two-sweep radar temporal window — the `_r4_tw2_`
   experiment lineage under `experiments/` is exactly that configuration — so its temporal
   support is comparable to v2. The mismatch is specifically against **Route B v1**, whose
   collector used a single sweep.
2. **Sample rate.** 5 Hz persisted (v1: 2 Hz), so an episode yields ~2.5× more frames and
   consecutive frames are more correlated. Splits are by complete episode, which keeps that
   correlation inside a split.
3. **Throughput bottleneck.** Prepared-input assembly averages ~0.30–0.40 s and now runs at
   10 Hz, i.e. 3–4 s of CPU per simulated second. It dominates wall-clock. Both hot loops are
   Python loops over ~37 k points per input: `rasterize_radar_channels` (the `legacy`
   rasterizer) and `StationaryTrackAccumulator.update`. `radar_fusion.py` already ships
   `rasterize_radar_channels_fast`, documented as an exact max-pool equivalent except for
   exact equal-magnitude signed-velocity ties. Switching to it would materially shorten the
   campaign but changes a data-producing code path, so it is raised as a decision, not
   applied.
4. **Extra warmup ticks.** The collector ticks the world 20 times before the route starts to
   observe and lock the camera cadence phase. Those ticks are not counted in the density
   runner's simulated duration.

---

## 7A. Population liveness fix (2026-08-25)

### 7A.1 The source correction

One function changed, in the canonical helper the Route B runner already imports:
`rl_agent/advisor_helper_scripts/codes/generate_traffic_v1.py` →
`TrafficPopulationManager._live_actor_map`.

```
- actors = self.world.get_actors(actor_ids)          # returns stale proxies
- return {a.id: a for a in actors if actor_is_alive(a)}
+ wanted = {i for i in actor_ids if i is not None}
+ snapshot = self.world.get_actors()                 # one fresh world snapshot
+ return {a.id: a for a in snapshot if a.id in wanted and actor_is_alive(a)}
```

SHA-256 `d68c377f…3c96` → `fba284df…3dcc`. The diff is confined to that method plus its
docstring; spawning, reconciliation, controllers and cleanup are untouched, the return type
(`dict[actor_id] -> actor`) is unchanged, and all seven call sites keep their contract. No
hash gate covers this file, and `run_route_b_density_loop.py` still matches its registered
SHA-256 `f2abd86c…5730`.

**`data_collection/generate_traffic_v11.py` was not integrated.** It is byte-identical to the
pre-fix canonical helper apart from one trailing blank line and therefore carried the same
liveness defect; adopting it would have changed nothing. Nothing about the advisor's
population-maintenance design needed replacing: loss detection, replenishment,
controller repair, orphan-controller tracking and cleanup were all already present and
correct. The single thing that was wrong was the **CARLA liveness query** those mechanisms
were built on, and that is all that was corrected. `TrafficPopulationManager` and
`PopulationLedger` remain the only population manager.

### 7A.2 Integration smoke — `POPULATION_FIX_PASSED`

`data_collection/route_b_perception_v2/population_liveness_smoke_v1.py`, fresh Epic CARLA,
8 managed walkers, one deliberate loss:

| Check | Result |
|---|---|
| Stabilized at target (8/8) | ✅ |
| Kill visible in world snapshot (7/8) | ✅ |
| **Exactly one loss recorded** (walker 49) | ✅ |
| **Exactly one replacement created** (walker 65) | ✅ |
| Population back to target (8/8 owned, 8/8 in world) | ✅ |
| Per-id lookup agrees with world snapshot | ✅ |
| Replacement controller ready **and** alive | ✅ |
| Cleanup leaves no actors (0 tracked survivors, 0 walkers left) | ✅ |
| World settings restored | ✅ |

Before the fix the same sequence produced zero detected losses and zero replacements.

---

## 7B. Radar radial-velocity sign convention (measured)

`data_collection/route_b_perception_v2/radar_velocity_sign_diagnostic_v1.py` drives an ego
forward and then in reverse through static geometry and compares boresight radar returns
(±4°) against the ego's own measured forward speed. Static geometry has no velocity of its
own, so the only relative motion is the ego's.

| Ego motion | samples | mean ego speed | mean radar velocity | fraction negative | ratio |
|---|---|---|---|---|---|
| Forward (closing) | 83 | +8.44 m/s | **−8.36 m/s** | **1.00** | −0.990 |
| Backward (receding) | 9 | −2.31 m/s | **+2.28 m/s** | **0.00** | −0.986 |

**Convention: CARLA radar velocity is the range rate — negative = closing (range
decreasing), positive = receding — and its magnitude equals the closing speed** (mean ratio
to ego speed −0.989 across all 92 samples). Raw buffer layout confirmed as
`(velocity, azimuth, altitude, depth)`, which `radar_raw_to_alt_az_depth_velocity` reorders
to `[altitude, azimuth, depth, velocity]`.

The first run of this diagnostic returned `UNDETERMINED` because it bucketed samples by the
*commanded* phase while the ego was still rolling forward through part of the reverse phase.
Bucketing by the ego's measured forward-speed sign resolves it; the underlying measurement
did not change.

---

## 7C. Runtime provenance for later radar-activity analysis

Added ahead of the Phase 2 rerun. **Raw runtime observables only** — no agent reducer, no
forward-angle thresholds, no radar-to-GT association, no actor IDs.

Per return, in `radar_points/<sample>.npz`, row-aligned with the existing arrays:

| Field | Meaning |
|---|---|
| `original_range_m` | depth **as measured** in that callback's own sensor frame |
| `original_azimuth_rad` | azimuth as measured, before motion compensation |
| `original_altitude_rad` | altitude as measured, before motion compensation |
| `radial_velocity_mps` | range rate; sign convention as measured in §7B |
| `observation_age_s` | prepared-frame radar timestamp − this return's callback timestamp |
| `sweep_offset` | 0 = current logical sweep, 1 = immediately previous |

`world_xyz`, `camera_xyz`, `velocity_mps`, `u`, `v`, `camera_depth_m`, `stationary_age_s`
and `valid_projection` are all retained unchanged and remain motion-compensated to the
prepared-frame radar pose. The npz also carries `prepared_timestamp_s`, `ego_speed_mps` and
`ego_velocity_mps`.

Per frame, in the new joinable sidecar `frame_runtime_provenance.csv` (`manifest.csv`'s
column list is fixed upstream in `common.py`, so it is not extended):
`sample_id, frame_id, prepared_timestamp_s, ego_speed_mps, ego_velocity_{x,y,z}_mps,
radar_window_returns, sweep_index`.

**Verified on real saved frames.** All six per-return arrays align row-for-row with
`world_xyz`/`u`/`velocity_mps` (34,850 returns); `sweep_offset` splits 17,418 current /
17,432 previous; `observation_age_s` takes exactly the four discrete values
{0.00, 0.05, 0.10, 0.15} s — the four callbacks of the 200 ms window. For current-callback
rows (age 0, no compensation applied) `original_range_m` matches the camera-derived distance
to a median 0.19 m / max 0.36 m, which is the camera↔radar mounting offset.

> **Observation, not a defect introduced here:** 0.043 % of returns (15 of 34,850) report a
> range beyond the configured 120 m, up to 172.7 m. Those same depth values already drove
> `world_xyz` and the radar tensor before this change; the rasterizer clamps the range score
> at `max_range_m`, so the tensor is unaffected. Flagged because a range-based activity
> analysis should not assume a hard 120 m ceiling.

---

## 7D. Rasterizer decision — **fast**, frozen for the campaign

### Comparison evidence

Offline comparison on **40 evenly spaced real saved frames** from the 30/30 smoke episode
(mean 37,105 returns/frame),
`data_collection/route_b_perception_v2/rasterizer_comparison_v1.py` →
`rasterizer_comparison_v1.json`:

| Channel | Differing elements (of 53,084,160) |
|---|---|
| occupancy | **0 — bit-identical** |
| radial_velocity | **0 — bit-identical** |
| stationary_age | **0 — bit-identical** |
| inverse_range | 446,680 (0.84 %) |

* Maximum absolute difference **5.960464477539063e-08** — approximately **half a float32
  ULP at 1.0**.
* **No equal-magnitude signed-velocity tie occurred on any frame**; the velocity channel is
  bit-identical, so the one documented behavioural difference of the fast path never
  materialized on real data.
* Runtime: legacy **0.2466 s/frame**, fast **0.0106 s/frame** → **23.3× isolated speedup**.

### Accepted numerical-equivalence tolerance

Recorded in `data_collection/configs/route_b_perception_v2.yaml`, in the collector as
`FAST_RASTERIZER_TOLERANCE`, and written into every episode's `metadata.json` and
`route_summary.json`:

```
bit_identical_channels: [occupancy, radial_velocity, stationary_age]
tolerant_channels:      [inverse_range]
max_abs_difference:     5.96e-08          # ~half a float32 ULP at 1.0
differing_element_fraction: 0.0084
equal_magnitude_velocity_ties_observed: 0
measured_speedup:       23.3
```

The rasterizer is now an explicit knob (`--rasterizer {fast,legacy}`, plumbed to the
unchanged `build_radar_sample(rasterizer=...)`), the canonical config resolves it to
**fast**, and the choice plus its tolerance is stamped into every episode manifest.
**Every canonical episode must use this same frozen fast rasterizer.**

### End-to-end fast smoke — `RADAR_SMOKE_PASSED` (19/19)

Fresh Epic CARLA, 45 s, `--rasterizer fast`
(`data_collection/route_b_perception_v2/fast_rasterizer_smoke_v1.json`). Every cadence,
alignment, tensor-shape, finite-value and cleanup gate is unchanged from the legacy smoke:

| | legacy smoke | fast smoke |
|---|---|---|
| World tick / sweeps / prepared / saved | 20.0 / 10.0 / 10.0 / 5.0 Hz | **identical** |
| Callbacks per sweep | exactly 2 | **exactly 2** |
| Window callbacks / span | 4 / 0.150 s | **4 / 0.150 s** |
| Window returns (mean) | 36,678.3 | **36,678.3** |
| Dropped / dup / out-of-order / reversed | 0 | **0** |
| Tensor failures | 0 | **0** |
| RGB+segmentation checks | 450/450 | **450/450** |
| Sensor cleanup / settings restored | clean | **clean** |
| **Total prepared-input wall time (mean)** | **0.2977 s** | **0.0567 s** |
| Total prepared-input wall time (max) | 0.3834 s | **0.1390 s** |

The headline number is the **total prepared-input time, not isolated rasterization**: the
full assembly — camera fetch, window construction, motion compensation, tracker update,
rasterization, validation — drops from 0.2977 s to **0.0567 s per prepared input, a 5.25×
reduction**. At 10 Hz that is ~3.0 s of CPU per simulated second down to ~0.57 s.

Radar tensors remain **uncompressed `.npy`**, as instructed.

### Raw range provenance is saved unmodified

CARLA occasionally reports `original_range_m` beyond the configured range — measured at
**0.043 % of returns (15 of 34,850), up to 172.7 m against a 120 m setting**. The saved raw
provenance is **not clamped or modified**. The documented downstream contract is that a
reducer treats only finite returns satisfying `0 < original_range_m <= configured_range_m`
as range-valid. This note is carried in the config, in each episode's `metadata.json`, and
here.

---

## 7E. 30/30 rerun crashed — SIGSEGV, unresolved

The post-fix traffic_30_30 rerun (`--rasterizer legacy`, fresh Epic CARLA) **did not
finish**. The client process terminated with **exit 139 (128 + 11 = SIGSEGV)** after
**1,380 saved frames** (~276 s of route, leg 16 of 19). No Python traceback was produced,
and because the crash bypassed the collector's `finally`, **no `route_summary.json` was
written** — so this episode has no gate report at all.

Preserved unmodified as
`20260825_smoke2_traffic_30_30_CRASHED_SIGSEGV_diagnostic_only/` (9.7 GiB, 1,380 frames).
Diagnostic evidence only; never a canonical episode.

What is known:

* **The CARLA server survived** the client crash and kept answering RPC.
* **Not memory pressure** — 62 GiB total, 14 GiB used, 47 GiB available at the time.
* `/proc/sys/kernel/core_pattern` routes cores to `apport`; no matching crash report was
  produced, and kernel logs are not readable from this account, so the faulting library was
  not identified.
* **The population fix was working at the moment of the crash**: the helper logged three
  `Detected population loss: vehicles=0 walkers=1` events, which the pre-fix code was
  structurally incapable of emitting. (`Replenished` is logged at INFO, below the root
  logger threshold, so its absence from the log is not evidence either way.)
* The immediately preceding legacy 30/30 episode, same route and density, completed all
  1,813 frames. The crash is therefore **non-deterministic**, not a hard regression.

Candidate contributing factors, none confirmed:

1. `notes.md:297` already records a CARLA SIGSEGV **stability ceiling on this hardware**
   under combined load at 30 NPC vehicles + 30 pedestrians, previously worked around by
   reducing resolution and density. The current configuration sits at that boundary, and the
   v2 contract **doubled camera render load** (free-running 20 Hz cameras instead of a
   commanded 10 Hz tick).
2. The population fix means walkers are now genuinely destroyed and respawned mid-episode.
   Actor churn during streaming is a known CARLA fragility, and this run is the first ever to
   exercise it on this route.
3. Per-tick client work was high: with the legacy rasterizer, prepared-input assembly
   averaged ~0.30 s at 10 Hz.

Factor 3 is materially reduced by the now-frozen fast rasterizer (§7D): total prepared-input
time falls 5.25× to 0.0567 s. That does not *explain* the crash, but it removes the largest
source of client-side per-tick load, and no canonical episode may use the legacy rasterizer
anyway.

**Consequence for sequencing.** There is currently **no passing traffic_30_30 episode**. The
instruction to proceed to traffic_50_50 after the fast smoke assumed the 30/30 had finished
and reported its gates; it crashed instead. 50/50 is a longer, denser episode, so it is
strictly more exposed to whatever caused this.

### Collector gap this exposed

A hard crash bypasses `write_summary`, so 1,380 saved frames exist with no summary, no gate
evaluation and no `frame_runtime_provenance.csv` / `collision_incident_windows.csv` (both are
written at episode end). A post-hoc summariser that can evaluate gates from an interrupted
episode directory would turn a crashed run into usable diagnostic evidence. Not implemented
— flagged.

---

## 7F. 30/30 fast rerun — `COLLECTION_SMOKE_FAILED`, root cause found

`20260825_smoke3_traffic_30_30_fast/`, fresh Epic CARLA, `--rasterizer fast`. Exit 2
(handled), `route_summary.json` **written**, 1,006 saved frames.

### What passed — the population fix is validated at scale

| | traffic_30_30 |
|---|---|
| Exact population at episode start | ✅ 30 / 30 |
| Vehicles alive | min **30**, mean **30.00**, 0 frames below floor |
| Pedestrians alive | min **29**, mean **29.98**, **0 frames below floor** |
| Max deficit span | **0.8 s** (limit `replenish_interval + 2` = 7 s) |

Compare the pre-fix episode: pedestrians decayed 30 → 21, mean 24.9, 1,707 frames below
floor, one 341 s deficit span. **The two gates that failed before now pass with margin**, and
six walker losses during this run were each detected and replenished.

The fast rasterizer also held up in real collection: total prepared-input wall time
**0.152 s** mean (vs **0.391 s** with legacy in the completed pre-fix episode). That figure
includes the camera fetch, so it is not directly comparable to the 0.0567 s isolated-smoke
number, which excludes it.

### What failed — the CARLA server died

```
failed gates: route_completed, no_collector_error, sensor_cleanup_succeeded,
              callbacks_per_sweep_exact, window_callbacks_exact,
              no_dropped_duplicate_or_reordered_callbacks
error: std::exception
```

All six are downstream of one event. `world.tick()` raised `std::exception`, the four sensors
could not be destroyed (`final_state: alive` with the server gone), and the CARLA server
process was **absent afterwards** with no fatal entry in its log. The tick stream had already
been degrading: effective world tick **19.71 Hz** instead of 20.00, with **59 skipped frames**
and ~38 logical sweeps holding one callback instead of two. In synchronous mode `world.tick()`
must advance exactly one frame, so frame skipping is itself a server-degradation signature,
not a collector fault.

### Root cause: mid-episode walker respawn starts AI controllers before a tick

Three 30/30 runs, one variable:

| Run | Rasterizer | Population fix | Walker respawns | Outcome |
|---|---|---|---|---|
| `20260824_smoke` | legacy | **no** | **0** | completed 1,813 frames |
| `20260825_smoke2` | legacy | yes | 3 | **client SIGSEGV** at 1,380 |
| `20260825_smoke3` | fast | yes | 6 | **server died** at 1,006 |

Both failures follow walker respawns closely — in run 3, three losses were logged between
saved frame 900 and 1000 and the run died at 1,006, roughly 1.2 s later.

The mechanism is in `TrafficPopulationManager._spawn_walkers`:

```python
self._spawn_walker_bodies_once(remaining)      # spawn bodies (apply_batch_sync)
self._spawn_missing_walker_controllers(...)    # attach controllers
self._initialize_walker_controllers(...)       # calls controller.start()
```

There is **no world tick between spawning a walker body and starting its AI controller**.
CARLA's own `PythonAPI/examples/generate_traffic.py` does exactly the opposite, with an
explicit comment:

```python
results = client.apply_batch_sync(batch, True)      # 3. spawn walker controllers
...
# wait for a tick to ensure client receives the last transform of the walkers we have just created
world.tick()
...
all_actors[i].start()                                # 5. only now
```

In synchronous mode a batch-spawned actor is not fully committed until the next tick, so
starting a controller on a body the server has not finished registering is unsafe.

**This path was unreachable before.** The liveness defect meant `missing_walkers` was always
0, so `reconcile()` never called `_spawn_walkers` with a non-zero count. Correcting
`_live_actor_map` made a second, pre-existing latent bug reachable for the first time — and
now it fires mid-drive-loop, with sensors attached, several times per episode. The population
liveness smoke did traverse it once in a small scene without sensors and survived, which is
consistent with a probabilistic corruption rather than a deterministic one.

### Smallest fix (proposed, not applied)

The population manager cannot tick the world itself: in synchronous mode the density
runner's drive loop owns the tick, and an extra tick inside `maintain_population` would
insert a world frame with no radar ingest, breaking the logical-sweep binning and the
runner's own simulated-time accounting.

Instead, **defer the controller start by one reconcile cycle**, using machinery that already
exists. In `_spawn_walkers`, spawn bodies and attach controllers but do not call
`_initialize_walker_controllers` on the new records, leaving `controller_ready = False`.
`reconcile()` already calls `_initialize_walker_controllers(self.walkers)` at the top of
every cycle, so the next cycle starts them — by which point many world ticks have elapsed.

* One-line-scale change confined to `_spawn_walkers`; no new machinery, no second population
  manager, no redesign of spawning, reconciliation, controllers or cleanup.
* Cost: a replacement walker stands still for up to one replenish interval (5 s) before its
  controller starts. It is alive, counted, and rendered throughout, so the population gates
  are unaffected.
* `spawn_initial_population` runs before the drive loop and is left alone.

This requires a second edit to the canonical helper, which is why it is proposed rather than
applied.

---

## 8. Storage extrapolation

Measured on the completed traffic_30_30 episode: **1,813 saved frames, 11.78 GiB**
(mean 6.98 MB/frame) for 362.5 s of route at 5 Hz.

| Corpus | Episodes | Est. frames | Est. storage |
|---|---|---|---|
| Per episode (traffic_30_30, measured) | 1 | 1,813 | **11.8 GiB** |
| Per episode (traffic_50_50, projected) | 1 | ~1,900–2,300 | **~13–15 GiB** |
| **Initial corpus** (bundles 1–3 × both densities) | **6** | ~11,000–12,500 | **~75–80 GiB** |
| **Final corpus** (+ bundle 4 × both densities) | **8** | ~15,000–16,500 | **~100–107 GiB** |

traffic_50_50 is projected upward because the denser scene slows the ego (more signal and
queue waiting), lengthening the episode; per-frame size is roughly density-independent since
the radar tensor is fixed-size and dominates.

Free space on this machine: **1.1 TiB**. The corpus fits comfortably, but see §2.4 — moving
`radar_tensors/*.npy` to `savez_compressed` would remove roughly 72 % of it losslessly.

Wall clock: ~33 min of driving per traffic_30_30 episode at RTF ≈ 0.185, plus world load.
Budget ~40–55 min per episode, so ~4–7 h for the six-episode campaign on a comparable
machine.

---

## 9. Terminal status

| Phase / step | Status |
|---|---|
| Phase 1 — radar-input correction and short smoke | **RADAR_SMOKE_PASSED** (19/19) |
| Population liveness source fix + integration smoke | **POPULATION_FIX_PASSED** (9/9) |
| Radar radial-velocity sign convention | **MEASURED** (negative = closing) |
| Rasterizer selection + fast end-to-end smoke | **RADAR_SMOKE_PASSED** (19/19), `fast` frozen |
| Phase 2 — collection smokes | **COLLECTION_SMOKE_FAILED** |
| Phase 3 — training preparation | **PREPARED** (nothing trained) |

**The population fix is validated and stays.** On the post-fix 30/30 episode the two gates
that previously failed now pass with margin: pedestrians min 29 / mean 29.98 / 0 frames below
floor / max deficit 0.8 s against a 7 s limit, vehicles exactly 30 throughout.

**Phase 2 still fails**, now for a different and fully diagnosed reason: correcting
`_live_actor_map` made a second pre-existing latent bug reachable — `_spawn_walkers` starts a
walker's AI controller in the same simulation frame that spawns its body, without the world
tick CARLA's own example documents as required. Both post-fix episodes died shortly after a
mid-episode walker respawn (§7F). A smallest fix is proposed but **not applied**, because it
is a second edit to the canonical helper.

**traffic_50_50 was not run.** No traffic_30_30 episode has passed, and 50/50 is longer and
denser, so it is strictly more exposed to the same defect.

### Episode inventory

| Directory | Rasterizer | Outcome | Canonical? |
|---|---|---|---|
| `20260824_smoke_traffic_30_30` | legacy | completed 1,813 frames; population gates failed | no — diagnostic only |
| `20260825_smoke2_..._CRASHED_SIGSEGV_diagnostic_only` | legacy | client SIGSEGV at 1,380; no summary written | no — diagnostic only |
| `20260825_smoke3_traffic_30_30_fast` | fast | server died at 1,006; population gates **passed** | no — diagnostic only |

All three are labelled `split: smoke`, none is a registered seed bundle, and none may enter
the canonical train/validation/test corpus. Every canonical episode must use the frozen fast
rasterizer.

### What was explicitly not done

No dataset campaign. No training step. No checkpoint written or overwritten. No locked-test
collection, evaluation or inspection. No OAI run. No second population manager. No
integration of `generate_traffic_v11.py`. Radar tensors remain uncompressed `.npy`. Saved raw
range provenance is unmodified and unclamped.

### Known gaps

1. **A hard crash bypasses `write_summary`.** The SIGSEGV run left 1,380 frames with no
   summary, no gate evaluation and neither sidecar CSV, since both are written at episode
   end. A post-hoc summariser over an interrupted episode directory would recover that.
2. **`route_summary.json → route_result` ledger fields are `null`.** `write_summary` runs in
   the drive wrapper's `finally`, before the density runner attaches `ledger.report()`.
   Population accounting in `route_summary.json` therefore comes from world-truth per-frame
   counts; the ledger numbers live in `route_metrics_summary.json`. The two disagreeing is
   exactly what exposed the liveness defect, so it is worth closing.
3. **Root cause of §7F is diagnosed but unproven.** The mechanism matches CARLA's documented
   requirement and the run-by-run correlation is 3/3, but it has not been confirmed by a
   controlled experiment.
