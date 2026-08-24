# Route B 30/30 perception-accuracy pilot — report

Date: 2026-08-24 EDT. Machine: current/main (RTX 5090 Laptop, CARLA 0.10.0, Town10HD_Opt).

> **Amended 2026-08-24 (second pass).** Sections 1-8 are the original collection record and
> are unchanged. A review decision then admitted the episode for offline perception analysis
> as `COMPLETE_WITH_COLLISION_PROVENANCE`; the evaluation was run and is reported in
> sections 9-13. Section 4 ("what was not produced") and the section 7 terminal are
> superseded by section 12 and are retained only as the record of the first pass.

| | |
|---|---|
| **Route-safety status** | **`FAIL` — unchanged.** 2 collision incidents. Not reclassified as a driving PASS. |
| **Analysis admission** | **`COMPLETE_WITH_COLLISION_PROVENANCE`** (review decision, second pass) |
| **Perception terminal** | **`CLEAR_ROUTE_B_COVERAGE_DEGRADATION`** |

The single Route B loop at 30 vehicles / 30 pedestrians failed the *route-safety* acceptance
rule: 2 collision incidents (one pedestrian, one NPC vehicle). Every other acceptance
condition held, the collisions did not interrupt collection or invalidate sensor
transforms, and all artifacts are complete — so the episode is usable as perception
evidence even though it is not usable as a driving result.

Evaluated on all 599 frames under the frozen contract, **both** the noAE primary and the
AE64 paired diagnostic degrade broadly against their retained historical references, with
ample eligible support (916 vehicle / 501 person GT). Excluding collision-adjacent frames
changes nothing material.

---

## 1. What ran

One fresh CARLA process, rendering enabled (`-quality-level=Epic`, windowed on `:0`),
one loop, no OAI, no training, no checkpoint write.

| Locked parameter | Value used |
|---|---|
| Map / route | `Town10HD_Opt`, qualified Route B full-map loop v1 (18 intermediate waypoints) |
| Loops | 1 |
| NPC vehicles / pedestrians | 30 / 30 (`--density traffic_30_30`) |
| Ego target speed | 25.0 km/h (passed explicitly) |
| Scenario seed / TM seed | 31 / 31 |
| Hybrid physics | **disabled** (`--no-hybrid-physics` passed explicitly) |
| NPC hardening / safe-vehicle filter | on / on (6 car blueprints kept, 5 excluded) |
| Lane offset | -0.5 m |
| Scenario interventions | disabled |
| Simulation | 20 Hz, `fixed_delta_seconds = 0.05` |
| RGB / semantic / depth / radar | `sensor_tick = 0.1` (10 Hz) |
| Radar | 200,000 points/s, 120 m range, 120°/30° FOV |
| Saved dataset | every 10th tick → 2 Hz |
| Dataset format | historical `scenesense_moving_ego_fusion_training_data.v1` |

Preflight `PASS`. Route JSON `fc4518a8…fd6e5`, progress CSV `97459385…d90c0`, density
runner `f2abd86c…5730` all hash-verified.

Command:

```bash
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
$PY data_collection/run_route_b_perception_collection.py \
  --density traffic_30_30 --scenario-seed 31 --tm-seed 31 \
  --target-speed-kph 25 --no-hybrid-physics \
  --output-dir fusion_training_data/route_b_perception_pilot_20260824_traffic_30_30/traffic_30_30_seed31_tm31_25kph
```

---

## 2. Collection acceptance — one condition failed

| Acceptance condition | Observed | Verdict |
|---|---|---|
| Full 19/19 waypoint route completed | 19/19, `all_ordered_waypoints_reached = true`, `completed = true` | PASS |
| B1/B2/B3 covered | `regions_covered = B1,B2,B3` | PASS |
| No watchdog abort | `watchdog_aborted = false`, `abort_reason = ""` | PASS |
| No scenario intervention | `intervention_count = 0`, `intervention_events = []` | PASS |
| **No collision incident** | **`collision_incident_count = 2`** (6 raw callbacks) | **FAIL** |
| Actor cleanup succeeded | `cleanup_succeeded = true`, 30/30 vehicles and 30/30 walkers alive at completion, 0 lost, 0 replenished | PASS |
| Perception-sensor cleanup succeeded | all 4 sensors `destroy_result: true`, `final_state: absent`, `cleanup_tick: ok`, 0 warnings | PASS |
| RGB/semantic/depth/radar frame- and timestamp-aligned | `max_timestamp_delta_s = 0.0`; exact per-sensor frame-ID equality enforced on every saved frame; max camera and radar transform displacement 0.0 m | PASS |
| Saved-frame interval ≈ 0.5 s | min = max = 0.5000000074505806 s across all 599 samples | PASS |
| New create-only output directory | created fresh; no existing artifact touched | PASS |

Runner terminal: `FAIL` (collision incidents). Route: 1251.69 m driven against a 1268.68 m
plan, 299.7 s simulated / 421.4 s wall, 5,994 ticks, 599 saved frames, loop closure 0.376 m
and 0.195°, 3 ego block events, 1 replan, 30 walker-brake ticks, 0 roadblock removals.
Episode size 3.4 GB (600-row `manifest.csv`, 6,941-row `object_boxes.csv`).

### The two collision incidents

**Incident 1 — pedestrian, episode t = 38.1–39.4 s, frames 4110–4136.**
Ego struck walker `94` (`walker.pedestrian.0031`) at world (98.70, 37.10) at **5.641 m/s**,
5 contact callbacks. `walker_braking_active = true` at contact.

Mechanism, traced from the episode's own saved GT (ego pose from `anchor_x/anchor_y`,
walker from `object_world_*`, offsets in the ego sensor frame):

```
frame   walker world       ego world        range    fwd     lat
 4058  (103.88, 38.84)   ( 86.47, 28.07)   18.96   15.74  10.56
 4068  (103.03, 38.86)   ( 89.15, 28.32)   15.93   13.10   9.07
 4078  (102.18, 38.87)   ( 92.33, 29.27)   12.18   10.61   5.99
 4088  (101.33, 38.89)   ( 95.17, 31.11)    8.25    7.75   2.82
 4098  (100.66, 38.90)   ( 97.27, 33.71)    4.43    4.41   0.38
 4108  (100.41, 38.91)   ( 98.60, 36.80)    1.34    0.87  -0.86
```

The walker is nearly stationary (x 103.9 → 100.4, y ≈ 38.9). The lateral offset collapses
from 10.56 m to inside the lane corridor in five saved frames because **the ego is turning
left** — its own pose swings from (86.47, 28.07) to (98.60, 36.80) as it turns north onto
the x ≈ 100 leg. The corridor rotates onto the pedestrian; the pedestrian does not walk
into the corridor.

This is a **different mechanism** from the walker-116 failure recorded in
`ROUTE_B_MPRIME_GENERALIZATION_REPORT.md` §3.4 (a genuine perpendicular late incursion).
`walker_ahead` gates on `agent._vehicle_obstacle_detected(walkers, reach_m)`, which tests
the **current** heading, so a pedestrian ahead-and-left of a vehicle about to turn left is
outside the corridor until the turn is already underway. Braking did engage (30 brake
ticks in the episode) but only inside the turn, at 5.641 m/s. Raising
`--walker-brake-distance-m` is unlikely to help here for the same reason it did not help
walker 116: the binding constraint is corridor *orientation during the turn*, not reach.

**Incident 2 — NPC vehicle, episode t = 264.65 s, frame 8641.**
Ego contacted vehicle `80` (`vehicle.mini.cooper`) at world (-75.77, 25.58) at **2.44 m/s**,
1 callback, `walker_braking_active = false`. The mini had come to rest at (-71.13, 24.65)
directly in the ego's path; the ego closed from 9.66 m to 3.86 m over three saved frames
while drifting from y = 24.43 to y = 25.73, and the contact registered with the NPC
2.55 m forward and 2.06 m to the ego's right — a low-speed clip while passing a stopped
NPC in the lane, with interventions disabled so the janitor did not remove it.

Both incidents are genuine simulator events, not instrumentation artefacts. Route
completion, sensor alignment and cleanup were unaffected.

### Why this diverged from the accepted 30/30 qualification

The same seed 31, 25 km/h, 30/30 configuration was run on 2026-08-24 **without perception
sensors and with hybrid physics on**
(`data_collection/route_b_density_validation/20260824_traffic_30_30_seed31_30kph/traffic_30_30_seed31_25kph.json`)
and was clean:

| | accepted density-only run | this perception pilot |
|---|---:|---:|
| Hybrid physics | on (runner default) | **off** (pilot requirement) |
| Perception sensors attached | no | 4 (RGB, semantic, depth, radar) |
| Simulated duration | 419.9 s | **299.7 s** |
| Ego block events | 11 | 3 |
| Walker brake ticks | 0 | 30 |
| Collision incidents | 0 | **2** |
| Terminal | `PEDESTRIAN_BRAKING_NOT_EXERCISED` | `FAIL` |

`--no-hybrid-physics` was a locked requirement of this pilot and **had not been exercised
in the accepted 30/30 qualification**. Disabling it puts all 30 NPC vehicles on full
physics regardless of distance, which changes NPC dynamics globally; the 120 s shorter
episode and the near-total change in blocking pattern are consistent with that. This is
stated as the most likely dominant cause, not as a proven one — a controlled A/B was not
authorized and was not run.

---

## 3. Object support that the episode did carry

Reported because it materially changes how a future Route B gate should be sized. Counted
under the frozen eligibility rule (actor GT, in front of and inside the 1280×720 camera
image, projected area ≥ 12 px, ≤ 40 m) — i.e. **40 m forward-camera eligible GT**, not
confirmed-visible GT, since reliable per-actor occlusion evidence is unavailable.

| | this 30/30 episode | historical low 5/5 | historical medium 15/15 | historical test split |
|---|---:|---:|---:|---:|
| Saved frames | 599 | 597 | 718 | 2,162 |
| Eligible vehicle GT | **916** (1.53/frame) | ~68 | ~425 | 2,468 |
| Eligible person GT | **501** (0.84/frame) | ~48 | ~252 | 1,431 |

Eligibility exclusions over 6,941 raw GT rows: 5,261 beyond 40 m, 263 below 12 px, 0
behind camera, 0 outside image, 0 non-actor, 0 missing geometry, **1,417 eligible**.

At 30/30 a single Route B loop supplies ~37% of the historical test split's vehicle
support and ~35% of its person support — roughly 10× the low-density episode. Pedestrian
support is no longer the limiting factor it was in the earlier gate. This is an
observation about scenario design, not a perception result.

---

## 4. What was not produced, and why *(first pass; superseded by sections 9-13)*

The pilot permits offline inference only on an accepted episode. Because collection
failed, the following were **not** written rather than populated with partial or
placeholder values:

- per-frame metrics CSV — not produced
- per-detection metrics CSV — not produced
- pooled / region / distance summary CSV — not produced
- noAE and AE64 metrics, and their deltas against the retained historical references — not produced

Both checkpoints were nevertheless hash-verified before collection and the result is
preserved:

| Model | Path | Required SHA-256 | Observed | Match |
|---|---|---|---|---|
| noAE (primary) | `experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt` | `f319e2a5…d4fa` | `f319e2a5…d4fa` | yes |
| AE64 (paired diagnostic) | `experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt` | `c6a2362c…9480` | `c6a2362c…9480` | yes |

The evaluator for the frozen contract is built, compile-checked and ready
(§6). It was never executed, so no evaluation provenance JSON exists beyond the
checkpoint-verification record.

Per the pilot instruction, the known pedestrian `object_speed_mps` / parked-label problem
was not fixed and no parked or moving/stationary accuracy is reported.

---

## 5. Code changes (minimal, as scoped)

`data_collection/run_route_b_perception_collection.py` only. The accepted Route B
coordinates, controller, traffic-light timing, density-runner behaviour, sensor geometry,
model architecture and checkpoints were **not** changed.

1. `DENSITIES` gains `traffic_30_30 = (30, 30)`. The historical `low`/`medium`/`dense`
   names are retained and commented as reproduction-only.
2. New `--target-speed-kph` (default 25.0), forwarded explicitly to the density runner.
3. New `--hybrid-physics` / `--no-hybrid-physics` pair, **defaulting to disabled** for this
   pilot; `--no-hybrid-physics` is appended to the runner argv when disabled. Passing
   `--hybrid-physics` restores the previous behaviour, so older episodes stay reproducible.
4. `target_speed_kph` and `hybrid_physics` are recorded in the preflight output,
   `metadata.json` (under `controller`) and `route_summary.json`.
5. The hardcoded seed check `(101, 1101)` became an allowlist `{(101,1101), (31,31)}`.
6. `scenario_id` now derives from the actual seeds instead of the hardcoded
   `seed101_tm1101` string, so metadata no longer mislabels a seed-31 episode.
7. `EXPECTED_RUNNER_SHA256` re-pinned from `59592ee8…49a4` to `f2abd86c…5730`.
   **The hash check was verified, not bypassed.** The density-runner diff was read in full
   and is exactly the reviewed 2026-08-24 population-ledger + `traffic_30_30`/`traffic_50_50`
   profile change documented in `ROUTE_B_TRAFFIC_30_50_CONFIGURATION.md` §3; the previous
   accepted value is retained in a comment above the constant.

New, versioned, create-only — the earlier AE64 evidence scripts in
`experiments/route_b_mprime_gate_20260824/` were **not** modified:

- `experiments/route_b_30_30_perception_pilot_20260824/run_route_b_eval_pilot_v1.py` —
  a copy of the gate evaluator parameterized on checkpoint / expected SHA-256 / expected AE
  bottleneck / a single episode directory, so noAE and AE64 run the identical frozen
  contract on the identical saved frames. Decoder and eligibility settings are unchanged
  from the gate script and from the retained historical `gate_eval` invocation for both
  checkpoints (`--object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120
  --match-distance-m 5.0 --max-gt-distance-m 40`, no quantization, no ROI threshold).
- `experiments/route_b_30_30_perception_pilot_20260824/summarize_route_b_eval_pilot_v1.py` —
  pooled / region / distance-band aggregation. The gate script's acceptance-floor table was
  **removed**: one temporally correlated episode cannot support a pass/fail threshold, so
  this emits descriptive metrics only.
- `experiments/route_b_30_30_perception_pilot_20260824/compare_to_historical_v1.py` —
  absolute deltas against the retained historical metrics JSON for each exact checkpoint,
  labelled as references, with no significance claim.

---

## 6. Exact output paths

Preserved episode (create-only, untouched since collection):

```
fusion_training_data/route_b_perception_pilot_20260824_traffic_30_30/traffic_30_30_seed31_tm31_25kph/
├── metadata.json                  # collection metadata (seeds, speed, hybrid_physics, sensors, weather)
├── route_summary.json             # route summary: sampling, sensor alignment, cleanup, per-frame counts
├── route_metrics.csv              # density-runner per-loop metrics
├── route_metrics_summary.json     # density-runner summary incl. collision incidents
├── manifest.csv                   # 599 saved frames
├── object_boxes.csv               # 6,941 GT rows
└── rgb/ semantic_tags/ masks/ radar_tensors/ radar_points/   # 3.4 GB of frames
```

Pilot evaluation tooling and provenance:

```
experiments/route_b_30_30_perception_pilot_20260824/
├── run_route_b_eval_pilot_v1.py
├── summarize_route_b_eval_pilot_v1.py
├── compare_to_historical_v1.py
└── checkpoint_verification.json   # both SHA-256 values verified; inference_executed = false
```

Collection log: `/tmp/route_b_pilot_30_30_collect.log`.
CARLA server log: `/tmp/carla_route_b_pilot_30_30.log`.

The commands the evaluator would run, once an accepted episode exists:

```bash
EP=fusion_training_data/route_b_perception_pilot_20260824_traffic_30_30/<accepted_episode>
OUT=experiments/route_b_30_30_perception_pilot_20260824/metrics

$PY experiments/route_b_30_30_perception_pilot_20260824/run_route_b_eval_pilot_v1.py \
  --dataset-dir "$EP" --profile traffic_30_30 --model-name noae \
  --checkpoint experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt \
  --expected-sha256 f319e2a5e8fb134e74c24c0822233e17368df6e4c733add658026603e131d4fa \
  --expected-ae-bottleneck 0 --output-dir "$OUT/noae" --device cuda

$PY experiments/route_b_30_30_perception_pilot_20260824/run_route_b_eval_pilot_v1.py \
  --dataset-dir "$EP" --profile traffic_30_30 --model-name ae64 \
  --checkpoint experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt \
  --expected-sha256 c6a2362c7c2d72ff31825508ae7532c0796ec063a8556317d47d8d30fad99480 \
  --expected-ae-bottleneck 64 --output-dir "$OUT/ae64" --device cuda

for m in noae ae64; do
  $PY experiments/route_b_30_30_perception_pilot_20260824/summarize_route_b_eval_pilot_v1.py \
    --metrics-dir "$OUT/$m"
done
$PY experiments/route_b_30_30_perception_pilot_20260824/compare_to_historical_v1.py \
  --metrics-root "$OUT" --out-csv "$OUT/historical_reference_deltas.csv"
```

---

## 7. Terminal and what it does and does not say *(first pass; superseded by section 12)*

**`INCONCLUSIVE`** — collection failed the acceptance rule. *(First-pass terminal, taken
before the episode was admitted for analysis. Superseded by section 12.)*

This says nothing about whether the historical dataset covers Route B. It is not evidence
for `NO_CLEAR_ROUTE_B_DEGRADATION_IN_ONE_EPISODE`, `AE_SPECIFIC_DEGRADATION` or
`CLEAR_ROUTE_B_COVERAGE_DEGRADATION`. No 50/50 confirmation is authorized on this basis,
and no dataset-reuse decision follows.

Notably, the failure is **not** the thin-pedestrian-support problem that sized the earlier
gate: this episode carried 916 eligible vehicle and 501 eligible person GT, ~10× the
low-density episode, which is ample for a descriptive pilot. The blocker is purely the
ego-controller/NPC-contact rule.

### Open decisions for the next authorization (not taken here)

1. **Whether `--no-hybrid-physics` should stay locked.** It is the one pilot requirement
   the accepted 30/30 qualification never exercised, and the episode it produced is
   materially different from the accepted one. If it is not needed for the perception
   question, dropping it returns the pilot to an already-qualified configuration.
2. **Whether an unavoidable NPC contact should remain blocking.** Both incidents are
   low-consequence in perception terms — 599 frames of correctly aligned, correctly
   cleaned-up data were collected around them, and neither corrupted a sensor record.
   Treating "contact while braking, below a speed threshold" as logged-but-non-blocking is
   a campaign-rule decision, not a code change. It is exactly the option
   `ROUTE_B_MPRIME_GENERALIZATION_REPORT.md` §5 already listed and left untaken.
3. **Turn-aware pedestrian braking.** Incident 1 is a corridor-orientation failure during
   a left turn, distinct from the §3.4 lateral-incursion failure. Both are outside the
   scope of any walker-brake-distance tuning. This is a genuine controller change requiring
   its own validation.

Options 1 and 2 are cheap and require no code; option 3 is real engineering. None was
attempted, because none was authorized.

---

## 8. Shutdown

CARLA was shut down cleanly after the episode. Verified: no `carla`/`unreal` process
remains, RPC port 2000 is free, and the only remaining GPU compute client is the unrelated
`gnome-remote-desktop-daemon`. In-simulation perception-sensor cleanup succeeded (all four
sensors `destroy_result: true`, `final_state: absent`), and the runner's own actor cleanup
reported `cleanup_succeeded = true` with 30/30 vehicles and 30/30 walkers still alive.

No model was trained, no checkpoint written or modified, no final-test evaluation run, no
OAI started, no production client or spatial-map code edited, and no additional CARLA
episode collected.

---
---

# Second pass — offline perception evaluation

Added 2026-08-24. No CARLA, no OAI, no collection, no training, no checkpoint write, no
controller or hybrid-physics change, no decoder tuning. Inference only, on the episode
already on disk.

## 9. Review decision and what it does not change

The episode is **still a route-safety `FAIL`** on 2 collision incidents. That status is
preserved verbatim in §2 and is not reclassified.

It is admitted for offline perception analysis as **`COMPLETE_WITH_COLLISION_PROVENANCE`**
on the grounds already established in §2: complete 19/19 route with B1/B2/B3 coverage; no
watchdog abort; no intervention; sensor/GT frame- and timestamp-alignment passed
(`max_timestamp_delta_s = 0.0`, exact per-sensor frame-ID equality on every saved frame,
0.0 m camera and radar transform displacement); actor and perception-sensor cleanup passed;
all artifacts complete; and neither collision interrupted collection or invalidated a
sensor transform.

This corrects the first pass's analysis-admission rule. It does not make the route a
driving PASS, and it does not retroactively make §7's `INCONCLUSIVE` wrong about *route
safety* — only about whether perception evidence could be extracted.

## 10. Evaluation as run

Both models were run through the already-prepared frozen contract, on the **identical 599
frames** and the identical eligible-GT slice, with the checkpoints and SHA-256 values from
§4 re-verified at startup:

- noAE primary: `f319e2a5…d4fa`, asserted `ae_bottleneck = 0` — matched.
- AE64 paired: `c6a2362c…9480`, asserted `ae_bottleneck = 64` — matched.
- Clean inference, no quantization, ROI drop q = 0, no predicted-world suppression, score
  threshold 0.20, image NMS radius 2 px, top-k 120, class-aware one-to-one world-XY match
  at 5 m, evaluation range ≤ 40 m, projected area ≥ 12 px, no lane-corridor filter, no
  per-model / per-region / per-distance tuning.

**Eligibility support**, identical for both models (the eligibility rule is model-independent):

| | count |
|---|---:|
| Raw GT rows | 6,941 |
| Excluded — beyond 40 m | 5,261 |
| Excluded — projected area < 12 px | 263 |
| Excluded — behind camera / outside image / non-actor GT / missing geometry | 0 / 0 / 0 / 0 |
| **Eligible** | **1,417** (916 vehicle, 501 person) |

This is "40 m forward-camera eligible GT", not confirmed-visible GT.

### 10.1 Pooled, all 599 frames — PRIMARY

| Metric | noAE all | noAE vehicle | noAE person | AE64 all | AE64 vehicle | AE64 person |
|---|---:|---:|---:|---:|---:|---:|
| Frames | 599 | 599 | 599 | 599 | 599 | 599 |
| Eligible GT | 1417 | 916 | 501 | 1417 | 916 | 501 |
| TP | 555 | 471 | 84 | 662 | 549 | 113 |
| FP | 741 | 527 | 214 | 1245 | 991 | 254 |
| FN | 862 | 445 | 417 | 755 | 367 | 388 |
| Precision | 0.4282 | 0.4719 | 0.2819 | 0.3471 | 0.3565 | 0.3079 |
| Recall | 0.3917 | 0.5142 | 0.1677 | 0.4672 | 0.5993 | 0.2255 |
| F1 | 0.4091 | 0.4922 | 0.2103 | 0.3983 | 0.4471 | 0.2604 |
| FP / frame | 1.2371 | 0.8798 | 0.3573 | 2.0785 | 1.6544 | 0.4240 |
| Duplicate-FP fraction | 0.4224 | 0.5446 | 0.1215 | 0.5205 | 0.6085 | 0.1772 |
| World-XY MAE (m) | 1.9501 | 1.8564 | 2.4754 | 1.7751 | 1.6879 | 2.1987 |
| World-XY RMSE (m) | 2.3126 | 2.2050 | 2.8416 | 2.1795 | 2.0966 | 2.5444 |
| 2D centroid error (px) | 64.84 | 73.32 | 17.29 | 27.34 | 11.70 | 103.31 |
| Length MAE (m) | 0.3875 | 0.4341 | 0.1264 | 0.3641 | 0.4136 | 0.1237 |
| Width MAE (m) | 0.1396 | 0.1526 | 0.0670 | 0.1321 | 0.1466 | 0.0616 |
| Height MAE (m) | 0.1388 | 0.1424 | 0.1184 | 0.1864 | 0.2025 | 0.1086 |

Segmentation (3-class confusion, pooled over all 599 frames):

| | background IoU | vehicle IoU | person IoU | foreground mIoU | mIoU (3-class) |
|---|---:|---:|---:|---:|---:|
| noAE | 0.9870 | 0.8204 | 0.2155 | 0.5179 | 0.6743 |
| AE64 | 0.9859 | 0.8027 | 0.2177 | 0.5102 | 0.6688 |

### 10.2 By region (descriptive)

| Region | model | veh P | veh R | veh XY (m) | veh n | per P | per R | per XY (m) | per n | FP/frame | fg mIoU |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 | noAE | 0.467 | 0.265 | 2.511 | 377 | 0.264 | 0.170 | 2.352 | 300 | 0.934 | 0.5257 |
| B1 | AE64 | 0.422 | 0.382 | 2.427 | 377 | 0.327 | 0.243 | 2.170 | 300 | 1.266 | 0.5055 |
| B2 | noAE | 0.179 | 0.500 | 2.739 | 54 | 0.426 | 0.202 | 3.007 | 114 | 1.131 | 0.4101 |
| B2 | AE64 | 0.142 | 0.704 | 2.088 | 54 | 0.342 | 0.219 | 2.342 | 114 | 2.022 | 0.4366 |
| B3 | noAE | 0.543 | 0.709 | 1.597 | 485 | 0.196 | 0.115 | 1.881 | 87 | 1.755 | 0.5203 |
| B3 | AE64 | 0.394 | 0.757 | 1.357 | 485 | 0.211 | 0.172 | 2.099 | 87 | 3.303 | 0.5194 |

B2 carries only 54 eligible vehicle GT; its vehicle figures are thin and should not be read
as a region effect on their own.

### 10.3 By distance band (descriptive)

| Band | model | veh P | veh R | veh XY (m) | veh n | per P | per R | per XY (m) | per n | FP/frame |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–10 m | noAE | 0.446 | 0.598 | 1.508 | 271 | 0.750 | 0.391 | 1.505 | 23 | 0.341 |
| 0–10 m | AE64 | 0.278 | 0.649 | **0.871** | 271 | 0.600 | 0.652 | 1.605 | 23 | 0.778 |
| 10–20 m | noAE | 0.541 | 0.776 | 1.715 | 272 | 0.292 | 0.366 | 2.219 | 142 | 0.509 |
| 10–20 m | AE64 | 0.450 | 0.879 | 1.632 | 272 | 0.313 | 0.366 | 2.189 | 142 | 0.678 |
| 20–30 m | noAE | 0.329 | 0.443 | 2.476 | 115 | 0.239 | 0.114 | 3.394 | 185 | 0.285 |
| 20–30 m | AE64 | 0.314 | 0.565 | 2.676 | 115 | 0.285 | 0.232 | 2.272 | 185 | 0.417 |
| 30–40 m | noAE | 0.522 | 0.182 | 3.021 | 258 | 0.100 | 0.013 | 3.859 | 151 | 0.102 |
| 30–40 m | AE64 | 0.406 | 0.267 | 3.033 | 258 | 0.120 | 0.020 | 4.291 | 151 | 0.205 |

Segmentation is not distance-resolved, so no IoU column appears here.

### 10.4 Absolute deltas vs the retained historical references

Full table: `experiments/route_b_30_30_perception_pilot_20260824/metrics/historical_reference_deltas.csv`.
The historical values come from the retained metrics JSON for each *exact* checkpoint, whose
generating command used identical decoder and eligibility settings. They are **references,
not statistically independent Route B acceptance thresholds** — one temporally correlated
episode of 599 frames cannot test them. Only absolute deltas are given; no significance and
no deployment approval is claimed.

| Metric | noAE pilot | noAE ref | Δ | AE64 pilot | AE64 ref | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Vehicle precision | 0.4719 | 0.5298 | −0.0579 | 0.3565 | 0.5014 | −0.1449 |
| Vehicle recall | 0.5142 | 0.8926 | **−0.3784** | 0.5993 | 0.9210 | **−0.3216** |
| Vehicle F1 | 0.4922 | 0.6650 | −0.1728 | 0.4471 | 0.6493 | −0.2023 |
| Vehicle XY MAE (m) | 1.8564 | 0.8759 | **+0.9805** | 1.6879 | 0.7717 | **+0.9162** |
| Person precision | 0.2819 | 0.6287 | −0.3469 | 0.3079 | 0.6289 | −0.3210 |
| Person recall | 0.1677 | 0.8532 | **−0.6856** | 0.2255 | 0.8644 | **−0.6389** |
| Person F1 | 0.2103 | 0.7240 | −0.5137 | 0.2604 | 0.7281 | −0.4677 |
| Person XY MAE (m) | 2.4754 | 1.0791 | **+1.3963** | 2.1987 | 1.0373 | **+1.1614** |
| Pooled recall | 0.3917 | 0.8782 | −0.4865 | 0.4672 | 0.9002 | −0.4330 |
| Pooled XY MAE (m) | 1.9501 | 0.9484 | +1.0017 | 1.7751 | 0.8653 | +0.9098 |
| Pooled XY RMSE (m) | 2.3126 | 1.2700 | +1.0426 | 2.1795 | 1.1695 | +1.0100 |
| Background IoU | 0.9870 | 0.9972 | −0.0103 | 0.9859 | 0.9966 | −0.0107 |
| Vehicle IoU | 0.8204 | 0.9313 | −0.1109 | 0.8027 | 0.9160 | −0.1132 |
| Person IoU | 0.2155 | 0.5897 | **−0.3743** | 0.2177 | 0.5625 | **−0.3448** |
| mIoU (3-class) | 0.6743 | 0.8394 | −0.1651 | 0.6688 | 0.8250 | −0.1562 |
| Dimension MAE (m) | 0.2220 | 0.1488 | +0.0731 | 0.2275 | 0.1483 | +0.0793 |

## 11. Sensitivity — excluding collision-adjacent frames

The flag is derived from the episode's own `route_metrics_summary.json` collision
incidents, widened by ±2.0 s in **both** recorded quantities (simulation timestamp, and
frame ID widened by 2.0 s × 20 Hz = 40 ticks); a saved frame is flagged if it falls in
either band. No sample-index guess is involved. Windows applied:

| Incident | sim-time band (s) | frame-ID band |
|---|---|---|
| 1 — walker 94 | 42.619 – 47.919 | 4070 – 4176 |
| 2 — vehicle 80 | 269.169 – 273.169 | 8601 – 8681 |

**18 of 599 frames (3.0%) were flagged**, identically for both models. The flag is written
as a `collision_adjacent` column on both the per-frame and per-detection CSVs, so both
scopes are aggregated from the same single inference pass — the model was not re-run.

| | noAE all-frames | noAE excl. | Δ | AE64 all-frames | AE64 excl. | Δ |
|---|---:|---:|---:|---:|---:|---:|
| Frames | 599 | 581 | −18 | 599 | 581 | −18 |
| Eligible GT | 1417 | 1366 | −51 | 1417 | 1366 | −51 |
| Vehicle recall | 0.5142 | 0.5242 | +0.0100 | 0.5993 | 0.6043 | +0.0050 |
| Vehicle precision | 0.4719 | 0.4716 | −0.0003 | 0.3565 | 0.3538 | −0.0027 |
| Vehicle XY MAE (m) | 1.8564 | 1.8572 | +0.0008 | 1.6879 | 1.6717 | −0.0162 |
| Person recall | 0.1677 | 0.1691 | +0.0014 | 0.2255 | 0.2296 | +0.0041 |
| Person precision | 0.2819 | 0.2746 | −0.0073 | 0.3079 | 0.3047 | −0.0032 |
| Person XY MAE (m) | 2.4754 | 2.5135 | +0.0381 | 2.1987 | 2.2182 | +0.0195 |
| FP / frame | 1.2371 | 1.2651 | +0.0280 | 2.0785 | 2.1170 | +0.0385 |
| Person IoU | 0.2155 | 0.2634 | +0.0479 | 0.2177 | 0.2699 | +0.0522 |
| mIoU (3-class) | 0.6743 | 0.6910 | +0.0167 | 0.6688 | 0.6869 | +0.0181 |

**The two brief incidents do not drive the metrics.** Every detection metric moves by less
than 0.02 except person XY MAE (+0.04 m). Person IoU is the largest mover at +0.05, and
even at 0.2634 / 0.2699 it remains ~0.30 below its historical reference. Excluding the
collision window does not change any conclusion in §12, and is **not** presented as the
primary result.

## 12. Interpretation

**Terminal: `CLEAR_ROUTE_B_COVERAGE_DEGRADATION`.**

Both noAE and AE64 degrade broadly, with adequate eligible support (916 vehicle / 501
person GT — comparable in magnitude to the historical test split's 2,468 / 1,431, from
~28% as many frames). The degradation is not confined to one class, one region or one
distance band, and it is not confined to precision.

Why the other terminals do not fit:

- Not `NO_CLEAR_ROUTE_B_DEGRADATION_IN_ONE_EPISODE`: recall, localization and person
  segmentation all move by large margins in both models (vehicle recall −0.32/−0.38,
  person recall −0.64/−0.69, XY MAE ~+0.9 to +1.4 m, person IoU −0.34/−0.37).
- Not `AE_SPECIFIC_DEGRADATION`: **noAE is not intact.** If anything AE64 has the *better*
  Route B recall of the two (vehicle 0.599 vs 0.514, person 0.226 vs 0.168), paid for with
  roughly double the false-positive rate (2.08 vs 1.24 FP/frame). The two models degrade
  together, which points at the data, not at the AE.
- Not `INCONCLUSIVE`: collection artifacts are complete and aligned, inference ran cleanly
  on both checkpoints with hashes verified, and pedestrian support (501 eligible GT, 388–417
  of them missed) is ample — this is the one earlier gate limitation that no longer applies.

**The known heatmap/decoder precision mechanism does not explain this.** That escape clause
applies only when recall, localization and segmentation stay comparable and most FPs are
duplicates. Here recall, localization *and* segmentation all degrade; and for the person
class the duplicate-FP fraction is only 0.12 (noAE) / 0.18 (AE64), so person precision loss
is mostly genuine false detections, not double-counting. Vehicle duplicates are substantial
(0.54 / 0.61) and do soften the vehicle precision deltas — which is precisely why vehicle
precision is the *smallest* delta in the table (−0.06 for noAE) while vehicle recall is one
of the largest.

### Evidence the evaluation itself is sound

Stated because a uniform pipeline fault would produce a similar-looking table:

- AE64 world-XY MAE at 0–10 m is **0.871 m**, against a 0.772 m historical pooled reference.
  A broken coordinate transform or camera matrix could not yield near-reference localization
  in the near band and 3.0 m in the far band; the error grows with range, as a genuine
  perception limit does.
- Background IoU is 0.986–0.987 against a 0.997 reference — segmentation is broadly intact;
  it is specifically the *person* class that collapses (0.22 vs 0.56–0.59).
- The two models were run on byte-identical frames with an identical, model-independent
  eligibility slice (1,417 eligible GT in both runs), so the noAE-vs-AE64 comparison is
  exactly paired.
- The eligibility audit independently reproduces the collection-time per-frame counts.

### Caveats that bound the strength of this result

1. **One episode, temporally correlated.** 599 frames at 2 Hz from a single loop are not
   599 independent samples. No confidence interval is quoted and none should be inferred.
2. **Distance mix differs from the reference.** From the retained historical per-detection
   CSVs, the true-positive GT distance mix (pinhole proxy `d ≈ f·gt_size_z/gt_bbox_h`,
   f = orig_w/(2·tan 60°)) puts only ~7.5% of historical vehicle GT in the 30–40 m band,
   against **28.2%** for Route B; for persons it is ~21% historical vs **30.1%** Route B.
   Route B is a harder, farther-out distribution, so part of the pooled recall gap is mix,
   not model. **This does not rescue the result:** even in the matched 0–10 m band, Route B
   recall is 0.598 / 0.649 (vehicle) and 0.391 / 0.652 (person) against historical *pooled*
   references of 0.89–0.92 and 0.85–0.86, and historical near-band recall would be at least
   as high as its pooled value. The gap survives distance matching.
3. **Historical per-band recall is not computable.** In the retained CSVs only `tp` rows
   carry GT geometry (`fn` rows have empty `gt_bbox_h`), so a like-for-like per-band recall
   comparison cannot be built from the artifacts on this machine, and the historical dataset
   itself is absent (`gate_eval/dataset` is a dangling symlink). The band comparison above
   is therefore Route B per-band against historical *pooled* — conservative, but not exact.
4. **The reference is a held-out split of the training distribution**; Route B is a
   different route. Some gap is expected by construction. The magnitude here (person recall
   0.17–0.23 against 0.85–0.86) is far beyond what that framing comfortably absorbs.
5. **The episode's route status is `FAIL`.** Perception evidence is admitted; driving
   evidence is not.

### Recommendation

Route B data collection is indicated, and per the terminal's own wording it should be
**targeted first**, before proposing full recollection. The pilot points at:

- **Beyond 20 m** — the sharpest fall for both models (person recall 0.11–0.23 at 20–30 m
  and 0.01–0.02 at 30–40 m). Note this is also where the distance-mix confound is largest,
  so targeted collection here doubles as the clean test of caveat 2.
- **Pedestrians generally** — 388–417 of 501 eligible persons missed, at every range and in
  every region.
- **B1** — the lowest vehicle recall of the three regions (0.265 noAE / 0.382 AE64) on solid
  support (377 eligible vehicle GT).

Not authorized by this result: any dataset-reuse decision, any 50/50 confirmation run, any
retraining, and any claim of statistical significance.

## 13. Second-pass outputs

```
experiments/route_b_30_30_perception_pilot_20260824/metrics/
├── historical_reference_deltas.csv                          # both models, absolute deltas
├── noae/
│   ├── route_b_per_frame_metrics.csv                        # 599 rows + collision_adjacent, timestamp_s
│   ├── route_b_per_detection_metrics.csv                    # + collision_adjacent, is_duplicate_fp
│   ├── route_b_density_region_summary.csv                   # PRIMARY, all 599 frames
│   ├── route_b_density_region_summary_no_collision_window.csv   # sensitivity, 581 frames
│   ├── route_b_summary.json / route_b_summary_no_collision_window.json
│   └── evaluation_provenance.json
└── ae64/   (same seven files)
```

`evaluation_provenance.json` for each model records `source_route_status = FAIL`,
`analysis_admission = COMPLETE_WITH_COLLISION_PROVENANCE`, `primary_scope = all_frames`,
`sensitivity_exclusion_window_s = 2.0`, both collision incidents verbatim with their derived
sim-time and frame-ID windows, `collision_flag_basis`, the checkpoint path with observed and
expected SHA-256, the asserted AE bottleneck, the full frozen `eval_settings`,
`quantization_mode = none_clean_inference`, `roi_drop_fraction = 0.0`,
`world_coordinate_suppression = false`, `predicted_world_suppression = false`,
`lane_corridor_filter = false`, `gt_slice = 40m_forward_camera_eligible_gt`, the eligibility
audit, and the 18 flagged sample IDs.

### Script adjustments (syntax-checked only)

- `run_route_b_eval_pilot_v1.py` — added `--route-metrics-summary`, `--collision-window-s`,
  `--source-route-status`, `--analysis-admission`; added `load_collision_windows` /
  `is_collision_adjacent`; added `collision_adjacent` and `timestamp_s` output columns; added
  the provenance fields above. No decoder, threshold, eligibility or matching change.
- `summarize_route_b_eval_pilot_v1.py` — added `--scope {all_frames,no_collision_window}`,
  which filters on the `collision_adjacent` column and suffixes its outputs. No metric
  definition change.

Both compile-clean (`py_compile`). The earlier AE64 evidence scripts in
`experiments/route_b_mprime_gate_20260824/` remain untouched, and the collected episode
directory was read-only throughout.
