# Next experiments — pick-up plan (written 2026-07-16; updated 2026-07-17)

**Exp 1 & 2 consolidated into agent guardrails → `rl_agent/AGENT_CONSTRAINTS.md`** (floor ~1.1 m; latency caps &
FPS floors per speed; Stage-1 master constraint `v·(Y_up+1/FPS) ≤ √(ε²−1.1²)`; next extension adds
`Y_down + Y_map_share`; speed-gated policy). **Experiment 3 is PARKED** (the FOV/parked-car diagnostic — resolved
that parked-ego is NOT the cause; the artificial single-target scene is; revisit another day with a training-like
scene).

**Status reconciled 2026-07-16 (after correcting the Experiment-3 protocol):**
- **Exp 2 (FPS staleness): SOUND** — corrected framing applied correctly, numbers physically right (v×1/FPS).
- **Exp 1 (curve plot): plots OK; the speed-confounded "curves are best" interpretation has been corrected.**
- **Exp 3 (FOV diagnostic): RESOLVED — the parked ego is NOT the cause; it's the artificial single-target SCENE.**
  Definitive offline test-set split by ego speed (200k, score 0.20): **stopped-ego veh-loc = 0.78 m, moving-ego =
  0.96 m** — the model localizes stopped-ego frames BETTER than moving (a static target is easier), and 33% of
  training is stopped-ego. So "parked is out-of-domain" was WRONG. The Exp3 1.5–2.2 m comes from the ARTIFICIAL
  scene: one isolated tagged vehicle at a fixed spawn spot, dead-behind, sparse — unlike natural cluttered Town10.
  **Fix Exp3 with a training-like scene** (target embedded in normal traffic, not a lone car at one spot), or
  investigate the specific spawn/placement. Split eval: `staleness/egospeed_split_ds/` + `staleness/egospeed_eval/`.
  A natural-scene post-hoc FOV split is now done in `staleness/fov_posthoc/`: on the offline 200k eval, FOV edge
  mainly hurts match availability and medium/far localization, not near-field localization. Do not claim a simple
  "center is always best" rule; use range-aware edge risk.
  Superseded lines below (kept for history):
- **Exp 3 (superseded): parked-ego underperforms (1.5 m@15 m / 2.2 m@10 m, scores ~0.25) vs a moving-ego control
  at the same 200k recipe (1.00 m, scores 0.565 ≈ offline 0.88 m). Likely mechanism: radar VELOCITY (Doppler) — a
  moving ego gives the static world ego-induced Doppler; a stopped ego gives ~zero Doppler.** BUT do not overstate as
  "out-of-domain": the training ego OBEYED lights (`ego_ignore_lights_pct=0`), so it DID sit stopped at reds — the
  stopped/zero-Doppler regime is **under-represented, not absent.** ⚠️ NOT yet cleanly isolated: the moving-ego control
  was *always moving* (ignore-lights) and used a *natural* scene, so 1.0 vs 2.2 m still conflates ego-motion with the
  Exp3 single-target/one-spot scene. An attempt to capture natural stopped-ego frames failed (ego cruised 50 s without
  hitting a red). **To settle it:** (a) capture stopped-ego-with-target frames reliably (spawn near a busy light / long
  run / queue in dense traffic) and compare loc on ego-stopped vs ego-moving frames (bin by `ego_speed_mps` in the
  metrics CSV), and/or (b) split the OFFLINE test set by ego speed to see if the model already localizes its own
  stopped-ego frames well. **Still: do NOT lower the score gate to 0.10 (cherry-picking).** Control runs:
  `...front_fusion_ego_25` (always-moving) and `...front_fusion_ego_117` (obeyed lights but never stopped).
  (Old text below is superseded.)
- **Exp 3 (superseded note): CENTERED ACCURACY GATE FAILED; no lateral run.** The corrected 15 m parked-ego test
  passed all 200k-PPS no-AE loopback, placement, camera-height, visibility, delivery, and radar checks, but produced
  **1.474/1.483 m** mean/median error at score >=0.20. A 10 m follow-up also passed protocol checks but failed the
  frozen score >=0.20 gate: only 20/60 matches within 2 m. Lower analysis thresholds recover accurate 10 m peaks,
  so the next issue is score calibration / duplicate peak selection, not visibility or radar support. See
  `experiment3_vehicle_lateral/README.md`.

## 2026-07-17 meeting action order — do these before agent training

The next work should proceed in this order. The goal is to turn the current capture-to-edge freshness constraint
into a full operational constraint that includes result return, queueing, delivery, and realistic channel behavior.

### Step 1 — Downlink latency and result-payload logging

Repeat a small Experiment-1/2-style run, but instrument the return path from the edge tail model back to the ego/car
display. This should answer how much budget is consumed after the edge result is ready.

Minimum timestamps/fields:

- `frame_id`
- `t_capture`
- `t_front_send`
- `t_edge_recv`
- `t_tail_done`
- `t_result_send`
- `t_car_result_recv`
- `t_display_ready`
- `uplink_payload_bytes`
- `downlink_payload_bytes`
- `result_type` (`boxes`, `seg_mask`, `overlay`, or combined)

Recommended first profiles:

1. Loopback no-AE, 200k radar recipe — debug baseline.
2. OAI no-AE u8 ROI0 — large-payload / latency-heavy case.
3. OAI AE-128 u4 ROI0 — current deployable compressed case.
4. Optional AE-128 u8 ROI0 — middle point if needed.

**2026-07-17 update:** Step-1 instrumentation is implemented in `staleness/carla_fusion_staleness_scenario.py`
and a clean experiment folder now exists at `downlink_latency_fps/`. The ideal-loopback FPS sweep completed under
batch `20260717_ideal_one_loop` using the no-AE 200k moving-ego recipe. Summary:

- delivery: 100% at 5/10/20/30 FPS;
- feature payload: ~1.08–1.09 MB, 19 UDP chunks;
- result payload: ~8–12 KB, 1 UDP chunk;
- capture/front-start → car result receive: ~88–90 ms p50;
- post-send feature-send-stamp → result-receive subpath: ~43 ms p50, ~54 ms p95;
- feature-upload payload-handling residual: ~31 ms p50; treat this as channel-independent payload handling
  (send burst/reassembly/deserialization/runtime), not RF/channel latency;
- result downlink/result-send to car receive: ~5 ms p50 for boxes/compact results;
- caveat: the training-route ego obeyed lights/traffic, so many frames are stopped; the route/autopilot did engage,
  but report speed distribution alongside latency.

Artifact: `downlink_latency_fps/IDEAL_LOOPBACK_RESULTS.md`.

**Bounded/default-buffer calibration:** the clean short 10 FPS / 100-frame rerun under
`net.core.rmem_max/wmem_max=212992` delivered only **3/100** frames with the no-AE 200k payload (~1.1 MB,
19 UDP chunks). This supersedes the first 1/100 calibration, which was useful diagnostically but had stale-port
and cleanup-trap hygiene issues. The bounded result confirms the old loopback setting is a UDP receive-buffer
reliability cliff, not a clean ~50 ms transport point for the current no-AE recipe. Treat it as Step-2
reliability/buffer evidence, not as a full Step-1 latency sweep. The high first-frame `back_ms` is a sparse-delivery
warm-up/outlier; later delivered frames returned to ~9–18 ms back compute. Artifact:
`downlink_latency_fps/BOUNDED_LOOPBACK_CALIBRATION.md`.

Post-cleanup validation: a stale local back-half on `51002` was stopped, the runner now fails fast when the back-half
port is already occupied, and a 100-frame post-cleanup ideal smoke (`smoke2_portclear_20260717`) reproduced the
same healthy ideal behavior (100/100 delivery, ~88 ms capture/front-start→result p50 estimate, ~43 ms post-send
subpath p50, ~5 ms tail/send-to-receive).

Pending: default OAI sweep. First OAI health check found the OAI core containers healthy, but `oaitun_ue1` was absent,
so UE/RAN/back-half bring-up is required before the OAI run.

This extends the Stage-1 constraint from `Y_up + 1/FPS` to the round-trip form:

> `L_total = Y_up + 1/FPS + Y_down + Y_map_share`

For now, `Y_map_share` can be a fixed placeholder. Do not claim final cooperative-perception compliance until
`Y_down` and map-sharing latency are measured.

### Step 2 — FPS × buffer size × reliability/delivery-rate experiment

After downlink logging is in place, test whether high requested FPS actually arrives fresh. Raw delivery count is
not enough: a frame that arrives after its speed-conditioned freshness deadline should be treated as stale.

Measure per run:

- generated frames
- queued frames
- sent frames
- edge-received frames
- tail-completed frames
- downlink-received frames
- displayed frames
- delivered FPS
- fresh-delivered FPS
- queue wait time
- end-to-end age
- dropped frames and `drop_reason`
- timeout/no-result rate

Suggested sweep:

| variable | values |
|---|---|
| FPS | 5, 10, 15, 20, 30 |
| buffer policy/size | latest-only/1, 2, 4, 8, 16 |
| payload profile | no-AE u8 ROI0, AE-128 u4 ROI0 |
| network mode | loopback, clean OAI, later Sionna-varying |

Key metric for the controller:

> `fresh_delivery_rate = frames displayed before their freshness deadline / frames generated`

This will tell us whether the guardrail should prefer latest-only dropping over stale queue accumulation for fast
objects.

### Step 3 — Experiment 3 / vehicle FOV measurement continuation

Keep this separate from the network guardrail work. The valid FOV evidence today is the natural-scene post-hoc split
in `staleness/fov_posthoc/`: edge position mainly hurts availability and medium/far localization, not near-field
localization. A controlled lateral sweep should only resume if the centered baseline passes under a deliberately
frozen threshold/target-selection rule in a training-like scene.

Stop/go rule before any controlled lateral sweep:

- center target visible and within 40 m;
- radar support and loopback delivery complete;
- score/confidence looks like the normal deployed regime;
- centered localization returns to the expected ~0.9–1.2 m range;
- threshold and target-selection rule are frozen before offsets begin.

If this gate fails again, report the controlled scene as an artificial-scene diagnostic and rely on the natural-scene
post-hoc FOV analysis for RL risk shaping.

### Step 4 — Sionna ray-traced channel realism

Do this after Steps 1–2 stabilize the logging schema. The Sionna work should reuse the same per-frame metrics, then
add position-dependent channel state:

- RSRP/SNR/SINR/path loss;
- LOS/NLOS/blockage regions;
- retransmission/HARQ proxies where available;
- delivery/drop/latency variation;
- buffer behavior under channel dips.

The intended output is not just a channel plot; it is a controller replay trace where the agent can see how realistic
channel variation changes payload feasibility, freshness, and reliability.

## 🚦 GUARDRAILS — read before running anything (this is the recurring failure mode)
A prior session shipped an Experiment-3 summary full of "findings" and "RL implications" built on **~46,000 m
localization errors and black camera frames**. Do not do this. Non-negotiable discipline:
1. **Validate the pipeline before interpreting.** Look at a real captured frame (not black); confirm the
   center/baseline loc error is in the sane ~1 m range. If a number is physically impossible (km-scale error,
   error that doesn't move with speed/latency as expected), STOP and fix the setup — do NOT write a results doc.
2. **Never rescue broken data with "relative patterns still hold."** If the absolute numbers are garbage, the
   relative pattern is garbage too.
3. **Reuse the validated components** (`carla_fusion_staleness_scenario.py`'s camera mount, radar build, and
   `decode_objects` call with `camera_matrix = actor_world_matrix(camera)`). Re-implementing the sensor/decode
   path from scratch is exactly what produced the 46 km bug.
4. **Watch for confounds before claiming an effect** (e.g., "curves are latency-tolerant" was just slow cars).
   Check per-cell sample sizes; a claim on n<15 or a physically implausible floor is noise, not a finding.

For a fresh coding session (keep context lean), start by reading the memory index (`MEMORY.md`) —
especially [[staleness_fps_results]], [[model_live_validation]], [[coop_fusion_findings]], [[rl_agent_pivot]] —
plus `staleness/STALENESS_RESULTS.md`. The three experiments came out of the 2026-07-16 supervisor meeting.

## Previously validated live setup (reference, not the final Experiment-3 recipe)
- **Sensor/model:** moving **car-height ego** (z=1.55, pitch −4°, fov 120° — matches training), RGB+radar fusion,
  no-AE u8 checkpoint `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`,
  loopback (clean, full delivery). Scenario: `staleness/carla_fusion_staleness_scenario.py`.
- **Important PPS distinction:** that 2026-07-15 live staleness validation happened to use **5,000 radar points/s**.
  The training/offline evaluation recipe and the selected deployment operating point use **200,000 points/s**.
  The 200k training manifest also uses radar HFOV 120°, raster radius 4, and a two-frame temporal max; the older live
  run used HFOV 100°, radius 2, and one frame. Experiment 3 uses the full training recipe, not just the PPS value.
- **GT convention (CRITICAL):** compare predictions against the actor ORIGIN — GT logs `origin_x/y` (= training's
  `actor.get_location()`), NOT bbox-center `world_x`. All staleness scripts already use `origin_*`.
- **Opportunity-window method:** ego drives among traffic; any vehicle in camera frustum AND ≤25 m matching a
  prediction (gate 2 m, score ≥0.2) is an observation, binned by MEASURED instantaneous speed.
- **High speeds (28–32 mph):** induced by over-speeding NPC traffic — `--npc-speed-difference-pct` negative
  (−45..−88) + `--npc-ignore-lights-pct 100`; still Town10 (in-domain); binned by measured speed.
- **Latency is swept ANALYTICALLY** on live captures: `error(Y) = ‖pred(t) − GT_origin(t+Y)‖` via `gt_at()`
  trajectory interp. Captures are live; the latency axis is post-hoc (no real transport delay injected).
- **Data on disk:** speed-sweep runs (walk→32 mph, 6 runs, run_group `speedsweep_*`) and FPS captures
  (`egofps_*`, `egofpsfast_*`) live under **`staleness/metrics_logs/scenesense_runs/`** (NOT the top-level one).
- **CARLA** on 127.0.0.1:2000 (Town10HD_Opt). venv: `carla_0_10_venv`. PYTHONPATH must include
  `pole_lraspp_multimodal_fusion:.:rl_agent/feature_ae`.

---
## Experiment 1 — COMPLETE: plots retained, interpretation corrected
**Result:** lowering the curve threshold to **4°/5 m** yielded 203 curve observations (curve plot exists:
`staleness/plots/roadstate_curve_speed.pdf`). ✅ The plots are fine.

The earlier claim that curves are more latency-tolerant was not supported; it was a speed-mix confound plus
small-sample noise and has now been removed from `STALENESS_RESULTS.md`. Evidence:
- The curve bin is **70% ~6 mph** (n=142/203); mid-speeds are essentially empty (**~10 mph n=1, ~14 mph n=2**).
  Cars *slow down on curves*, so the curve bin is dominated by slow traffic → the *pooled* curve curve looks
  latency-tolerant simply because it's mostly slow cars, NOT because curvature helps.
- At matched speed the curve cells are thin (~18 mph n=24, ~28–32 mph n=21) and show a physically implausible
  *lower floor* (1.05 m vs straight 1.24 m at Y=0) — that's noise, not a real curvature benefit.
- **Correct framing:** report the road-state split as **straight vs intersection** (well-sampled); note curves
  are under-sampled and speed-confounded, so we can't claim a curvature effect without targeting curved roads at
  controlled speeds. The earlier curvature-aids-prediction hypothesis is withdrawn.

Original protocol retained for provenance:

Context: `staleness/make_roadstate_speed_plots.py` splits the per-speed error(Y) into straight vs intersection
(plots `roadstate_{straight,intersection}_speed.pdf`). Town10 curves came up ~0 because the curve threshold
(`CURVE_DEG_PER_5M = 8.0`) is strict / curves sit inside junctions.
- **Do:** lower `CURVE_DEG_PER_5M` (try 3–5°/5 m) and re-run on existing speed-sweep data; if a "curve" bin gets
  enough samples (≥~15), add a third per-speed plot. If still too sparse → target a curved road segment
  (spawn ego on a known Town10 curve) in a short capture.
- **Output:** `roadstate_curve_speed.pdf` (if feasible) + a note in STALENESS_RESULTS.md Result 1b.

## Experiment 2 — COMPLETE: FPS as spatial-map staleness

**Result:** `staleness/make_fps_speed_report.py` computes the worst-case held-map error at
`t + 1/FPS` and the coupled lag `Y_up + 1/FPS` from the existing speed sweep. At ~32 mph, zero-network-
latency error drops from **15.02 m at 1 FPS** to **2.02 m at 10 FPS**, **1.52 m at 20 FPS**, and
**1.41 m at 30 FPS**. With AE-128 latency (105 ms), the corresponding 32-mph values are about
3.0 m at 10 FPS, 1.7 m at 20 FPS, and 1.5 m at 30 FPS. Main output:
`staleness/plots/fps_mapStaleness_worstcase.pdf`; see Result 2 in `staleness/STALENESS_RESULTS.md`.

Original corrected protocol retained for provenance:

> We previously did single-frame vs Kalman accumulation. **That is NOT what the supervisor wants — drop it.**
> The point: the **spatial map** queries the latest detection; between frames it holds a STALE position. At
> update rate FPS, worst case the held position is a full **1/FPS** old (queried just before the next frame).
> Static car → fine; fast car → error ≈ `v × (1/FPS)` **even at ZERO network latency**. FPS is a latency analog.

- **Metric:** at 0 network latency, map-staleness error = `‖pred(t) − GT_origin(t + 1/FPS)‖` (worst case),
  and/or `t + 1/(2·FPS)` (average query time). Bin by measured speed.
- **Sweep:** FPS ∈ {1, 5, 10, 15, 20, 25, 30} × speed 0–32 mph. **Computable POST-HOC from the existing
  speed-sweep captures** — no new runs (detection accuracy is FPS-independent; the FPS effect is pure staleness,
  and `gt_at()` extrapolates the trajectory for the 1/FPS lookahead, which is exactly the v·(1/FPS) displacement).
- **Plot:** loc error (y) vs object speed (x), **one line per FPS** — shows error flat for static, exploding at
  low FPS for fast cars, collapsing toward the model floor (~1.1 m) at high FPS. Worst-case line is the headline;
  optionally add the average-case.
- **Then:** repeat with added network latency `Y_up` → error at lag `Y_up + 1/FPS` (FPS+latency coupling). Overlay the
  real operating Y (loopback ~50, AE-128 ~105, no-AE ~267 ms).
- **Build from:** `staleness/make_speed_error_report.py` / `make_roadstate_speed_plots.py` (same collection loop;
  just set the lookahead lag = 1/FPS instead of a fixed Y, and make FPS the line variable).

## Experiment 3 — Radar/camera FOV-position diagnostic (the parked-car localization puzzle)
Why: loc error was ~2 m even for a nearby/parked car; segmentation looked OK but localization was off. Hypothesis:
accuracy degrades as the target moves from FOV center toward the edge / partial occlusion.

### ⚠️ A FIRST ATTEMPT WAS BROKEN AND HAS BEEN DELETED — don't recreate its mistakes
The prior `carla_fov_position_diagnostic.py` + `staleness/fov_diagnostic/` (results, black frames, false summaries)
produced ~46,000 m "errors" yet still shipped "findings" + RL implications. **All of it was deleted** (script,
folder, and the `fov_*.pdf` plots). Start fresh. The three bugs to NOT repeat:
1. **Camera mounted inside the car mesh** → black frames. It used `cam_loc = Location(x=0, y=0, z=0.5)` relative
   to the ego (car centre, 0.5 m up = inside the cabin). MUST use the validated ego-camera mount: **x=1.8, z=1.55,
   pitch=−4°, fov=120°** (see `carla_fusion_staleness_scenario.py` DEFAULT_EGO_CAMERA_X/Z).
2. **`decode_objects` got the intrinsics, not the extrinsics** → world coords exploded to ~46 km. It called
   `decode_objects(..., camera_matrix=camera_intrinsics)`. `camera_matrix` MUST be the camera's **world matrix**
   `actor_world_matrix(camera)` (sensor-relative xyz → world). Intrinsics are ONLY for the 2D pixel projection.
3. **Ego placed off-road / misaligned** — spawned at `TARGET.x − 15, yaw=0` regardless of the road. Place the ego
   ON the lane behind the target: `wp = map.get_waypoint(target_loc)`, step back along the lane, use the lane
   heading as yaw. Sweep lateral offset **within the lane/road**, not into buildings.

### ROOT CAUSES FOUND; V2 HARDENED AS A DIAGNOSTIC HARNESS — use only after baseline parity
The "90–100 m error / impossible coordinates" was **NOT a coordinate-transform bug** (that was a red herring the
prior session chased). The first audit fixed two real bugs:
1. **GT target position read as (0,0,0)** — `target.get_location()` was called BEFORE a `world.tick()`, so the
   actor transform hadn't initialized; every error was measured against the origin (~100 m away). **Fix: tick
   after spawn, then read.**
2. **Ego & target not co-located** — they were placed at two unrelated map spawn points. **Fix: anchor the ego
   on a lane, place the target `EGO_DISTANCE_M=15` m ahead on that lane (`ego_wp.next(15)`), assert separation ≈15 m.**
Also: set `ClearNoon` weather (matches training — the "blurry sky" was a domain mismatch); `try_spawn_actor` +
skip for extreme offsets that land off-road (was crashing with `std::exception`). It now completes with sane
coordinate magnitudes, but that single-frame output is still **diagnostic-only**, not a valid result.

A second parity audit found and fixed four more deviations from the validated staleness pipeline:

1. CARLA's BGRA image was channel-swapped before `prepare_fusion_input`, causing the model to receive BGR.
2. `decode_outputs` received `(width,height)` instead of the required `(height,width)`.
3. The radar raster used 1536×864 display intrinsics while being built at 768×432 model resolution.
4. The radar was mounted at `(0,0,0)` and a stale warm-up sample was consumed; it now uses the validated
   `(x=2,z=1)` mount, exact camera/radar frame synchronization, and a persistent stationary tracker.

The hardened V2 also defaults to 15 frames/offset, records misses separately using a 5 m target-association gate,
logs both raw radar points inside the target box and the model's radar-support score, saves annotated sanity frames,
and enforces a center-error sanity gate. `analyze_fov_diagnostic.py` reports conditional localization error and
target match rate separately so edge failures cannot be hidden by selection bias.

### Exploratory FOV runs rejected by the baseline-parity audit (2026-07-16)

The hardened script fixed genuine frame, color, synchronization, transform, and spawn bugs, so it remains useful as
a development harness. However, its recent static/convoy outputs are **not Experiment-3 evidence** and must not be
cited. The audit found that they:

- ran the checkpoint directly in one process instead of exercising the deployed split **loopback + uint8 + zlib +
  ROI 0** path;
- used **5,000 radar points/s**, whereas the checkpoint's training/offline test samples and selected operating point
  use **200,000 points/s**;
- used a **5 m association gate**, which can label a materially inaccurate prediction as a target match; at the
  stricter 2 m gate, the static center had only 10/30 matches and the moving center 40/60;
- proceeded to angle/scene sweeps even though the centered errors were about 2 m, above the expected controlled
  no-AE floor.

The discarded folders are useful only as a record of harness development; their numerical FOV pattern, replication,
and moving/static comparisons are withdrawn. The scripts remain, but no new sweep should run until the following
baseline gate passes.

### Corrected centered baseline (2026-07-16) — accuracy gate FAILED

The removed moving-ego/lead-NPC convoy was not the agreed Experiment-3 protocol and none of its measurements are
evidence for this experiment. Do not recreate or cite it.

The canonical corrected run is `experiment3_vehicle_lateral/centered_200k_15m_v1/`. Before collection, the harness
was fixed to apply the training collector's 30 physics-settling ticks; without that step the parked ego remained at
the elevated CARLA spawn transform and raised the camera from the training `~1.57 m` world height to `~2.30 m`.
The retained run measured camera world `z=1.565 m`, exact 15.000/0.000 m forward/lateral placement, 60/60 target
visibility and loopback delivery, and raw target radar support in all frames (mean 1,860 points).

At score ≥0.20 with the no-AE evaluation decoder (NMS radius 2, top-k 120), 59/60 opportunities matched within
2 m. Their mean/median/p90 error was **1.474/1.483/1.569 m**. All 60 matched within 5 m, with
**1.506/1.483/1.577 m** mean/median/p90. The result is much closer than the rejected harness outputs but still
does not reproduce the accepted floor, so it is a stop result, not permission to relax the gate.

### 10 m centered follow-up (2026-07-16) — frozen score gate still FAILED

To test whether the centered failure was mostly distance, the same corrected harness was rerun with the tagged NPC
at 10 m forward and 0 m lateral. The run is `experiment3_vehicle_lateral/centered_200k_10m_v1/`. It again passed
protocol checks: 60/60 loopback results, 60/60 visible target opportunities, exact 10.000/0.000 m placement,
camera world `z=1.565 m`, and raw radar support in every frame (mean 3,354 points).

At the frozen score ≥0.20 analysis threshold, however, only 52/60 frames had a score-qualified vehicle prediction
and only 20/60 matched within 2 m. The ≤2 m conditional mean/median/p90 was **1.239/1.251/1.336 m**, but
availability and median still failed. Within 5 m, 52/60 matched with **2.215/2.657/3.016 m** mean/median/p90.

The useful clue: lower-score target peaks are often more accurate than the score ≥0.20 candidate. A read-only
threshold diagnostic found:

- score ≥0.05: 60/60 within 2 m; 0.984/1.026/1.079 m mean/median/p90.
- score ≥0.10: 60/60 within 2 m; 1.106/1.113/1.284 m mean/median/p90.
- score ≥0.15: 57/60 within 2 m; 1.216/1.194/1.590 m mean/median/p90.
- score ≥0.20: 20/60 within 2 m; 2.215/2.657/2.992 m mean/median/p90.

So 10 m is not permission to run the lateral sweep under the old gate. It suggests the next controlled decision is
whether to re-freeze the analysis threshold / target-selection rule before any 10 m lateral run.

### Remaining work — stop/go sequence

- [x] Freeze the recipe: parked car-height ego at spawn 80, one tagged Lincoln 15 m ahead at **0 m lateral
  offset**, object-origin GT, distance ≤40 m, current no-AE checkpoint, and real loopback/u8/zlib/ROI-0 transport.
- [x] Collect exactly 60 measured centered frames after sensor/target warm-up, with **200k PPS**, HFOV 120°, radar
  raster radius 4, and a two-frame temporal max.
- [x] Inspect RGB/annotated frames and verify target identity, placement, visibility, range, lateral offset, radar
  support, complete loopback delivery, and GT/prediction coordinate magnitudes before calculating a result.
- [x] At score ≥0.2, report all 60 target opportunities, misses, 2 m and 5 m association availability, and
  conditional mean/median/p90 localization error. **Gate failed: 1.474/1.483 m mean/median at ≤2 m.**
- [x] Try 10 m center as a closer-distance sanity check. **Frozen score ≥0.20 gate still failed**: only 20/60
  matches within 2 m, though lower analysis thresholds reveal accurate lower-score peaks.
- [ ] Only after a future centered gate passes under an explicitly frozen threshold/target-selection rule, move the
  NPC laterally through negative and positive offsets while the ego, forward depth, target yaw, sensor recipe,
  checkpoint, and transport remain fixed. Log signed lateral offset and pixel-x for every frame; repeat center at
  the end to check drift.
- [ ] Keep distance as a separate later variable (10/20/30/40 m if needed); do not mix distance and lateral sweeps
  initially and do not analyze targets beyond 40 m.

### Correct approach — REUSE the validated components, don't reimplement the sensor/decode path
- Strongly prefer adapting **`carla_fusion_staleness_scenario.py`** (its ego-camera mount, radar build, and
  `decode_objects` call with `camera_matrix = actor_world_matrix(camera)` are already correct and validated) OR
  the radar-diagnostic pattern `carla_radar_pedestrian_distance_pps_diagnostic.py`. The from-scratch rewrite is
  what introduced both bugs.
- **Baseline setup:** parked car-height ego at spawn 80 with one tagged NPC 15 m ahead at zero lateral offset;
  200k PPS; real loopback/u8/zlib/ROI-0 path. RGB+radar → model → target localization error against actor-origin GT.
- **Sweep (only after the baseline passes):** keep the ego parked and kinematically translate only the NPC through
  negative and positive lateral offsets. Hold forward depth and vehicle yaw fixed; log actual range, signed offset,
  pixel-x-from-center, visibility, and radar support so the small range change is explicit.
- **Plot:** target availability and localization error vs signed FOV angle / pixel position, with misses separate.

### 🚦 MANDATORY sanity checks BEFORE writing any findings (this is the whole point)
1. **Look at a captured frame** — it must show the target car clearly, NOT black / not the ego's own body.
2. **Center-offset loc error should reproduce the no-AE baseline before any sweep.** The old automatic 3 m center
   gate is too permissive. For discussion, use mean ≤1.3 m and median ≤1.2 m at 15 m as provisional stop/go bounds,
   with match availability reported separately; freeze the final bounds before collection.
3. Only once (1) and (2) pass, run the sweep and report. If a result contradicts physics, say so — never paper
   over it with "relative patterns still valid."

**Downstream (only if valid):** if center detections are more accurate, the agent can prioritize center-FOV
tensors under bandwidth limits. Also separate localization-head error vs radar-support/occlusion (log both).

---
## Also carry forward
- **RL agent state documentation — complete:** object/ego dynamics and map freshness are now explicit in
  `SCENESENSE_RL_SCHEMA.md` and `rl_agent/REQUIREMENTS_AND_RL_DESIGN.md`. Wiring those fields into an
  executable controller remains open.
- **OAI config sweep (paused):** findings in `oai_config_sweep/OAI_CONFIG_FINDINGS.md` — config barely moves
  transport in single-UE; compression is the lever. Gotcha: automated rfsim gNB↔UE restarts are flaky → use a
  full cold-restart per config; extreme UL TDD not achievable (gNB K2 + UE DCI limits). Revisit under multi-UE /
  channel impairment (Sionna-RT).
- **Meeting framing that landed:** cars tracked ≤25 m (median 13 m); high speeds via NPC over-speed; latency swept
  analytically on live captures. The speed/latency trend persists on straight roads and at intersections;
  intersections are worse at 267 ms. Curves remain under-sampled and speed-confounded. The zero-latency floor
  stays near 1.1–1.2 m.
