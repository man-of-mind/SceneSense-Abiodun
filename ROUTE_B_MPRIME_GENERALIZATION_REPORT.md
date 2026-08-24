# Route B M-prime (AE64) Generalization Gate — Report

Date: 2026-08-24 EDT
Machine: current/main (RTX 5090 Laptop, CARLA 0.10.0)
Campaign terminal state: **`STOPPED_AFTER_MEDIUM_COLLISION`** (both campaigns)
Verdict: **`INCONCLUSIVE`**

The minimal Phase-1 sensor-cleanup fix is complete and validated in two live episodes.
Phase 2 stopped at the medium density on a genuine ego/pedestrian collision, which is an
explicit stop condition. Because the three-density Route B corpus was therefore never
completed, **Phase 3 was not run** and no model-generalization metrics exist. This report
records what was established, what blocked, and what is required to unblock. No perception
number in this document is a measured Route B result, because none were produced.

---

## 1. Model and interface verification (completed before CARLA startup)

| Item | Value |
|---|---|
| Checkpoint | `experiments/ae_integrated_20260710/ae64/checkpoints/ae64_integrated/best.pt` |
| SHA-256 observed | `c6a2362c7c2d72ff31825508ae7532c0796ec063a8556317d47d8d30fad99480` |
| SHA-256 expected | `c6a2362c7c2d72ff31825508ae7532c0796ec063a8556317d47d8d30fad99480` — **match** |
| AE family / bottleneck | `ae_arch=v2`, `ae_bottleneck=64` (integrated, runs inside `model.forward`) |
| Model task | `segmentation_plus_learned_object_localization` |
| Input size | 768 x 432 |
| Object classes | `("vehicle", "person")`, `predict_bbox2d=true`, `head_arch=shared`, `head_depth=3` |
| Warm start | `experiments/mprime_dropaware_20260708/stage2_obj_drop/checkpoints/mprime_stage2_obj_drop/best.pt` (M-prime) |

The common warm-start checkpoint was **not** substituted, and no other AE family was
loaded. The evaluator asserts `ae_bottleneck == 64` and re-verifies the SHA-256 at
startup, so a silent substitution fails closed.

### Historical comparability — provenance confirmed

The retained historical metrics for this exact checkpoint are
`experiments/ae_integrated_20260710/ae64/gate_eval/metrics/test_fusion_evaluation_metrics.json`.
Its generating command is recorded at `rl_agent/ae_integrated/run_ae_integrated.sh:33`:

```
--object-score-threshold 0.20 --object-nms-radius-px 2 --topk-objects 120 \
--match-distance-m 5.0 --max-gt-distance-m 40 --device cuda
```

with no `--quantization-mode` and no `--roi-threshold`, i.e. clean inference with the AE
inside `forward`. These are **identical** to the settings required for this gate, so the
retained numbers are directly comparable and were adopted as the comparison baseline:

| Historical (test split, 2,162 samples) | Vehicle | Person |
|---|---:|---:|
| Precision | 0.5014 | 0.6289 |
| Recall | 0.9210 | 0.8644 |
| World-XY MAE (m) | 0.7717 | 1.0373 |
| TP / FP / FN | 2273 / 2260 / 195 | 1237 / 730 / 194 |

Pooled: XY MAE 0.8653 m, XY RMSE 1.1695 m, dimension MAE 0.1483 m, FP/frame 1.383,
mIoU (3-class) 0.8250, vehicle IoU 0.9160, person IoU 0.5625, background IoU 0.9966.

The normal-service floors supplied for this gate coincide with these retained values, so
the gate is a like-for-like comparison rather than a new threshold.

**Caveat on availability:** the historical dataset
(`fusion_training_data/moving_ego_pps200000_merged_8loops_stride2`) is **not present on
this machine** — the `gate_eval/dataset` symlink is dangling. Comparison would therefore
have been against the retained metrics artifacts, not a re-run. No missing historical
dimension or IoU value was fabricated.

---

## 2. Phase 1 — minimal sensor-cleanup fix (COMPLETE, validated live)

### Root cause

`LegacyPerceptionCollector.stop_sensors` verified sensor liveness immediately after
issuing `destroy()`. CARLA applies destruction server-side on the **next tick**, and the
episode's synchronous world was never ticked between the destroy calls and the
`is_alive` check. Every correctly destroyed sensor therefore still read as alive, and the
postcondition reported failure unconditionally. The pilot's
`STOPPED_AFTER_LOW_CLEANUP_FAILURE` was this false negative, not a leaked actor.

Confirmed empirically: in both episodes run here, all four sensors returned
`destroy_result: true` and, after one cleanup tick, `final_state: absent`.

### Change (only the cleanup postcondition)

`data_collection/run_route_b_perception_collection.py`:

1. Per-sensor record of sensor type (`type_id`), actor ID, stop result/exception,
   destroy return/exception, and final actor-liveness result.
2. All four listeners stopped, then all four sensors destroyed (reverse spawn order).
3. The synchronous world is advanced by exactly **one cleanup tick**, with any tick
   exception captured.
4. Success is determined by final actor absence:
   - destroy returned `False` but the actor is confirmed absent → **warning only**;
   - actor still alive, or final verification unavailable (including a failed cleanup
     tick) → **failure**.
   A stop exception on a confirmed-absent actor is likewise a warning, not a failure.
5. The concise per-sensor result is saved in `route_summary.json` under `sensor_cleanup`
   (`succeeded`, `cleanup_tick`, `warnings`, `sensors[]`). The pre-existing
   `sensor_cleanup_succeeded` key is retained for backward compatibility.

Route B, controller behaviour, sensor placement, sensor configuration, GT generation,
sampling and dataset format are **unchanged**. The accepted Route B JSON, progress CSV and
qualified density runner are untouched and still hash-match (preflight `PASS`).

### Observed result (all four episodes)

```
perception sensor cleanup OK: rgb#66=absent, semantic#67=absent, depth#68=absent, radar#69=absent   (low)
perception sensor cleanup OK: rgb#141=absent, semantic#142=absent, depth#143=absent, radar#144=absent (medium)
```

`succeeded: true`, `cleanup_tick: "ok"`, `warnings: []` in both. The fix is validated.

---

## 3. Phase 2 — Route B episodes

Two campaigns were run. §3.1-§3.2 cover the first (walker brake distance at the qualified
10.0 m); §3.4 covers the second (raised to 20.0 m under an authorized scope decision).
Four episodes were collected in total; dense was never reached in either.

### 3.1-3.2 First campaign (walker brake distance 10.0 m)

Fixed settings, identical across episodes: Town10HD_Opt; accepted Route B JSON
(`fc4518a8…fd6e5`) and progress CSV (`97459385…d90c0`); qualified runner
(`59592ee8…49a4`); scenario seed 101; Traffic Manager seed 1101; lane offset -0.5 m;
walker detection 10 m; qualified NPC hardening and safe-vehicle filtering; interventions
disabled; fresh-world qualified default weather; 20 Hz simulation with every tenth tick
saved; a new create-only output directory per episode. The server was launched with an
explicit `-quality-level=Epic` (CARLA's default for this build, and the renderer locked
for primary design rows).

| Density | Requested V/P | Terminal | Frames | Sim / wall | Driven | Loop closure | Collisions | Interventions | Watchdog | Cleanup |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| low | 5/5 | **PASS** | 597 | 298.8 s / 339.3 s | 1251.6 m | 0.52 m / 1.53° | 0 | 0 | no | OK |
| medium | 15/15 | **FAIL — collision** | 718 | 359.2 s / 439.5 s | 1252.0 m | 0.46 m / 0.30° | 1 incident (47 contacts) | 0 | no | OK |
| dense | 25/25 | **NOT_RUN** | — | — | — | — | — | — | — | — |

### Low — PASS

Sampling interval constant at 0.50000000745 s (min = max). Maximum cross-sensor timestamp
spread 0.0 s; maximum camera and radar transform displacement 0.0 m. 597 manifest rows,
597 saved frames, 1,076 object-GT rows. Route completed, 19/19 waypoints, zero collisions,
zero interventions, no watchdog abort. 3.4 GB.

The runner's own terminal status was `PEDESTRIAN_BRAKING_NOT_EXERCISED`
(`walker_brake_ticks = 0`), which is why the process exit code was 1. That status is a
scenario-richness observation — the ego never had to brake for a walker — and is **not**
one of the specified stop conditions (`run_route_b_density_loop.py:869`). It is
distinct from `FAIL` and `INTERVENED` in the same function. Low is therefore treated as a
passing episode, with the caveat in §3.3.

Reproducibility versus the earlier pilot is close at the route level but **not
bit-identical**: 1251.62 m driven here vs 1251.69 m in the pilot, 597 saved frames in
both, but **5 replans here vs 3 in the pilot** under the same 101/1101 seed bundle. The
episode is therefore statistically repeatable, not deterministic — see §3.3.

### Medium — FAIL (genuine collision, blocking)

The ego struck walker `116` (`walker.pedestrian.0037`) at episode time 33.75 s
(frame 13765) at world (85.84, 28.07) — on the B1 eastbound y~28 artery — at
2.651 m/s with a first-contact impulse norm of 85.73. The contact persisted for 2.7 s of
simulation as the ego ground to a halt against the walker, producing 47 collision
callbacks that the runner correctly folded into **one** collision incident (last contact
frame 13819, 36.45 s).

`walker_braking_active` was `true` throughout, and the episode logged 81 walker-brake
ticks: the qualified controller **did** detect the pedestrian and was braking, but the
10 m walker detection distance with interventions disabled was insufficient to avoid
contact. Everything else in the episode was clean — route completed, 1252.0 m driven,
0.46 m loop closure, zero interventions, no watchdog abort, perfect sensor alignment,
constant 0.5 s sampling, and successful sensor cleanup.

Per the stop-on-first-failure rule the campaign halted immediately; **dense was never
started** and its output directory does not exist. The medium output must remain
failure-tagged and must not be admitted to training or evaluation.

**Whether this reproduces is genuinely open.** The seed bundle is fixed at 101/1101, but
this campaign shows the pipeline is not bit-deterministic under a fixed seed: the low
episode diverged from the pilot in replan count (5 vs 3) while matching route length to
0.07 m. Asynchronous physics/timing jitter is enough to move a marginal
vehicle/pedestrian interaction across the contact threshold. A second medium run under
the identical contract could therefore pass or fail, and this was **not** verified — a
repeat medium run was not authorized under the stop-on-first-failure rule.

### 3.3 Reproducibility

Every episode reached **19/19 ordered waypoints** with `all_ordered_waypoints_reached =
true` and `regions_covered = B1,B2,B3`. Route length is highly repeatable across all four
episodes collected here (1251.58-1251.99 m against a 1268.68 m planned route).

Minor event counters do vary run to run under the fixed 101/1101 seed bundle — the low
episode logged 5 replans in the first campaign, 2 in the second and 3 in the earlier
pilot. **The collision does not.** As §3.4 shows, it recurred with identical actor ID,
identical position, identical episode time and identical contact speed across two
campaigns with different controller settings. Replan counts are sensitive to blocked-ego
timing; the collision event is deterministic.

### 3.4 Second campaign: walker brake distance raised to 20 m (authorized scope change)

After the first campaign stopped, the walker detection/brake distance was raised from the
qualified 10.0 m to **20.0 m** under an explicit scope decision, and **all three**
densities were re-collected in a new campaign directory
(`route_b_perception_gate_20260824_111201_EDT_wb20`). Every density was recollected, not
just medium, because the walker brake distance is a controller parameter shared by all
episodes: mixing low at 10 m with medium at 20 m would confound density against controller
configuration, which is exactly the comparison this gate exists to isolate.

`--walker-brake-distance-m` is now an explicit CLI flag on
`run_route_b_perception_collection.py`, **defaulting to the qualified 10.0 m** so the
original contract is unchanged unless overridden. The value used is recorded in both
`metadata.json` and `route_summary.json`, so every episode is self-describing.

| Density | Result | Frames | Brake ticks | Driven | Collisions |
|---|---|---:|---:|---:|---:|
| low | **PASS** | 597 | 0 | 1251.58 m | 0 |
| medium | **FAIL — same collision** | 614 | 114 | 1251.82 m | 1 incident (49 contacts) |
| dense | **NOT_RUN** | — | — | — | — |

**The change did not help, and the reason is now established.** The collision recurred
with:

| | 10 m campaign | 20 m campaign |
|---|---|---|
| Walker actor ID | 116 | 116 |
| Position | (85.84, 28.07) | (85.84, 28.07) |
| Episode time | 33.75 s | 33.75 s |
| Ego contact speed | 2.651 m/s | 2.651 m/s |
| `walker_braking_active` | true | true |
| Episode brake ticks | 81 | 114 |

Doubling the reach produced 41% more braking across the episode as expected, yet the
contact speed at the incident was **bit-identical**. That is the signature of a collision
whose geometry is unaffected by detection range.

#### Mechanism: late lateral incursion, not insufficient detection range

Tracing walker 116's recorded GT against ego pose through the approach (forward/lateral
offsets are in the ego sensor frame):

```
frame   walker world      ego world        range    fwd     lat
9228   ( 87.28, 19.98)  ( 69.80, 27.98)   19.23 m  15.60  -8.20
9268   ( 87.60, 23.06)  ( 69.80, 27.98)   18.47 m  15.95  -5.12
9308   ( 87.89, 26.12)  ( 75.37, 28.02)   12.66 m  10.71  -1.99
9318   ( 87.98, 26.89)  ( 78.70, 28.04)    9.35 m   7.49  -1.20
9328   ( 88.19, 27.64)  ( 82.10, 28.05)    6.11 m   4.33  -0.43
9338   ( 88.47, 28.20)  ( 85.09, 28.06)    3.38 m   1.61  +0.12
```

The walker holds x ~ 87.3-88.5 and moves steadily in +y from 19.98 to 28.27 — it is
**crossing the carriageway perpendicular** to the ego's eastbound travel along y ~ 28.
It is in sensor range the whole time (19.23 m at first trace point, inside even the 10 m
reach shortly after), but its lateral offset only closes from -8.20 m to inside the ego's
lane corridor (|lat| < ~1 m) at frame 9328-9338 — by which point it is just **4.33 m then
1.61 m** ahead.

`walker_ahead` gates on `agent._vehicle_obstacle_detected(walkers, reach_m)`, which flags
only actors already inside the ego's forward corridor. A pedestrian that enters that
corridor 4 m ahead cannot be braked for in time at any reach value, which is precisely why
10 m and 20 m produce the identical 2.651 m/s contact. **Raising the parameter further
(30 m, 40 m) is contraindicated by this evidence** — it would add spurious braking
elsewhere on the route without touching this failure.

Avoiding this contact requires a different detector, not a larger number: braking on
predicted crossing intent, i.e. using the walker's lateral velocity *toward* the corridor
rather than its current presence *inside* it. That is a genuine controller-logic change,
well beyond the "minimal cleanup fix" scope of this task, and was not attempted.

### 3.5 Side observation — pedestrian GT speed is almost always zero

Not part of the gate, but surfaced while tracing the collision and worth recording.
In the medium 20 m episode, `object_speed_mps` is exactly `0.00` for **99.4%** of
pedestrian GT rows (1428/1436, max 1.69 m/s), while the position trace above shows
walker 116 covering ~0.77 m per 0.5 s saved frame (~1.5 m/s) with a reported speed of
0.00 throughout. Vehicles are 81.2% zero-speed, which is more plausible given the stalled
NPCs, but still worth a look.

Because `stationary_label` and `parked_label` are derived from this field, essentially
every moving pedestrian is likely labelled stationary/parked. This does **not** invalidate
the primary detection/localization/segmentation metrics this gate targets — the parked
head is a secondary output — but it would matter for any parked-classification claim or
velocity-conditioned logic. Flagged for separate triage; not investigated further here.

### 3.6 Eligible-GT support is thin at these densities (relevant to gate design)

Counting occurrences across saved frames under the primary eligibility rule
(in front of camera, inside the original image, box area >= 12 px, distance <= 40 m):

| Density | Frames | Eligible vehicle GT / frame | Eligible person GT / frame | Approx. total eligible V / P |
|---|---:|---:|---:|---:|
| low | 597 | 0.114 (max 3) | 0.080 (max 2) | ~68 / ~48 |
| medium | 718 | 0.592 (max 4) | 0.351 (max 2) | ~425 / ~252 |

For comparison the historical test split carried 2,468 eligible vehicle GT and 1,431
eligible person GT. A single low-density Route B episode supplies roughly 3% of that
support, so a person-recall gate on low alone would carry very wide confidence intervals.
This is a property of Route B (a wide-open full-map loop, where the 40 m / 12 px / FOV
gate is strict), not a defect, but it matters for how any future Route B gate is sized.

---

## 4. Phase 3 — offline model evaluation: **NOT RUN**

The instruction sequences Phase 3 after all three episodes pass collection. Two of the
three did not exist in a passing state, so the evaluation was not executed and **no
Route B perception metrics were produced**.

Consequently the following deliverables contain no data and were deliberately not
written rather than populated with placeholder or partial values:

- per-frame metrics CSV — not produced
- density/region summary CSV — not produced

Fabricating either, or presenting a low-only run as the three-density gate, would
misrepresent the evidence.

### Tooling is built, verified and ready

Both scripts are complete, compile-checked, and reuse the canonical decoder and
eligibility functions from `pole_lraspp_multimodal_fusion` unchanged, so results will be
directly comparable to the retained historical numbers:

- `experiments/route_b_mprime_gate_20260824/run_route_b_eval.py`
- `experiments/route_b_mprime_gate_20260824/summarize_route_b_eval.py`

They implement: SHA-256 + AE64 bottleneck assertion at startup; clean inference (no
quantization, no ROI/feature drop with q=0, no world-coordinate suppression); score
threshold 0.20, image NMS radius 2 px, top-k 120; class-aware one-to-one greedy world-XY
matching at 5 m (`greedy_match_predictions`); the 40 m operating-range gate applied to
predictions as well as GT; per-frame 3x3 segmentation confusion; and per-detection rows
carrying density, Route B region, distance band, world-XY error, 2D centroid error
(rescaled from head resolution to original image pixels), and per-axis length/width/height
absolute error.

Reported scopes would be pooled, per density, per region, per density x region, and by
distance band (0-10, 10-20, 20-30, 30-40 m), with identical decoder and eligibility
settings for every density and region and no per-density threshold tuning.

Region assignment: each frame is assigned to the nearest of the 18 ordered Route B
intermediate waypoints by ego world XY, and that waypoint's label from the route JSON's
`route_b_regions` array (B1 central city corridor, B2 southern regions, B3 northern and
outer extents).

GT eligibility is audited with first-failing-reason attribution, reporting counts for:
behind the camera, outside the original image, box area below 12 px, and beyond 40 m,
plus non-actor GT source and missing geometry. No lane-corridor or "ego path" filter is
applied — pedestrians approaching from the side remain eligible.

### Per-actor visibility / occlusion evidence is **not available**

The auxiliary "confirmed visible" slice cannot be computed from this corpus, and was not
approximated. Two independent reasons, both verified in the collection code:

1. The depth camera is spawned only as a synchronisation witness. Its frames are used
   solely in the cross-sensor timestamp check and are **never written to disk**
   (`run_route_b_perception_collection.py`, `save_sample_files`).
2. `instance_raw_path` does not contain instance IDs. `semantic_tags_from_image`
   returns the red channel of the semantic-segmentation frame, i.e. the **class tag
   only** (`carla_collect_parked_ego_fusion_training_data.py:242`).

A class-level mask overlap test was considered and rejected: it cannot separate two
overlapping same-class actors, so it is not reliable per-actor visibility evidence. The
primary historical 40 m / FOV evaluation would in any case have remained the primary
result.

---

## 5. Decision

**`INCONCLUSIVE`**

This is a collection-side blocker, not a model finding. Across two campaigns and four
episodes, no Route B perception metric was measured, so there is no evidence for or
against AE64 M-prime generalization, and no basis to prefer targeted hard-negative
collection, targeted dense collection, or full recollection. Specifically:

- the low-density precision/background question is untested;
- the dense-only recall/localization question is untested (dense was never collected);
- no Route B region or density comparison exists.

Choosing any of `REUSE_EXISTING_DATASET_FOR_AE64_PILOT`,
`TARGETED_ROUTE_B_DATA_NEEDED` or `FULL_ROUTE_B_RECOLLECTION_JUSTIFIED` would require
inventing the measurements that the stop rule prevented from being taken.

### What is required to unblock

The blocker is a **deterministic, seeded pedestrian crossing** at (85.84, 28.07) on the
B1 corridor at episode time 33.75 s, where walker 116 enters the ego's lane corridor only
~4 m ahead of the moving ego (see §3.4).

**Already tried and falsified:** raising the walker brake distance from 10 m to 20 m. It
increased episode braking by 41% and changed the contact speed by exactly zero, because
detection range is not the binding constraint. Raising it further is contraindicated by
the same evidence.

Remaining options:

1. **Relax the stop rule for unavoidable NPC incursions.** The ego behaved correctly here
   - it detected the walker, emergency-braked, and was down to 2.651 m/s and still
   decelerating at contact. CARLA's walker AI is known to walk into slow and stopped
   vehicles. Treating "contact while `walker_braking_active` is true, below some speed
   threshold" as a logged-but-non-blocking event would let collection proceed on the
   unmodified qualified controller. This is a campaign-rule decision, not a code change.
2. **Change the Traffic Manager / scenario seed** for medium and dense so this specific
   crossing does not occur. Cheap, but it only moves the dice - other crossings may
   collide - and it breaks the fixed 101/1101 contract.
3. **Implement crossing-intent braking**: brake on a walker's lateral velocity *toward*
   the corridor rather than its presence *inside* it. This is the actual engineering fix
   and would likely make Route B collection robust at all densities, but it is a genuine
   controller-logic change requiring its own validation.
4. **Accept low-density-only evidence**, understanding that its ~68 vehicle / ~48 person
   eligible GT cannot support the stated recall floors at useful precision, and that it
   cannot answer the dense-density question the gate exists to ask.

Note that dense (25 pedestrians) has still never been collected in either campaign. Under
options 2 and 3 it remains unproven; under option 1 it becomes collectable immediately.


Only the walker-brake-distance change was authorized, and it is now falsified.

---

## 6. Exact commands and output paths

Server:

```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping
./CarlaUnreal.sh -RenderOffScreen -nosound -carla-rpc-port=2000 -quality-level=Epic
```

Preflight and episodes (run from
`/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun`,
with `PYTHONPATH` unset):

```bash
PY=/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python
CAMP=fusion_training_data/route_b_perception_gate_20260824_104927_EDT

$PY data_collection/run_route_b_perception_collection.py --density low \
  --output-dir "$CAMP/low_5_5_seed101_tm1101" --preflight-only

$PY data_collection/run_route_b_perception_collection.py --density low \
  --output-dir "$CAMP/low_5_5_seed101_tm1101"

$PY data_collection/run_route_b_perception_collection.py --density medium \
  --output-dir "$CAMP/medium_15_15_seed101_tm1101"
```

Second campaign, walker brake distance raised to 20 m (all three densities requested):

```bash
CAMP2=fusion_training_data/route_b_perception_gate_20260824_111201_EDT_wb20

for den in low medium dense; do
  $PY data_collection/run_route_b_perception_collection.py --density $den \
    --walker-brake-distance-m 20.0 --output-dir "$CAMP2/${den}_..._seed101_tm1101"
done
```

Dense was never invoked in either campaign.

Outputs created:

| Path | Size | State |
|---|---|---|
| `..._gate_20260824_104927_EDT/low_5_5_seed101_tm1101` | 3.4 GB | PASS (walker brake 10 m) |
| `..._gate_20260824_104927_EDT/medium_15_15_seed101_tm1101` | 4.1 GB | **FAIL - collision, do not admit** |
| `..._gate_20260824_111201_EDT_wb20/low_5_5_seed101_tm1101` | 3.4 GB | PASS (walker brake 20 m) |
| `..._gate_20260824_111201_EDT_wb20/medium_15_15_seed101_tm1101` | 3.5 GB | **FAIL - same collision, do not admit** |

Not created (neither campaign):

- `..._gate_20260824_104927_EDT/dense_25_25_seed101_tm1101`
- `..._gate_20260824_111201_EDT_wb20/dense_25_25_seed101_tm1101`


Evaluation tooling (built, not executed):

- `experiments/route_b_mprime_gate_20260824/run_route_b_eval.py`
- `experiments/route_b_mprime_gate_20260824/summarize_route_b_eval.py`

The command they would run, once a qualified three-density corpus exists:

```bash
$PY experiments/route_b_mprime_gate_20260824/run_route_b_eval.py \
  --campaign-dir "$CAMP" \
  --output-dir experiments/route_b_mprime_gate_20260824/metrics --device cuda

$PY experiments/route_b_mprime_gate_20260824/summarize_route_b_eval.py \
  --metrics-dir experiments/route_b_mprime_gate_20260824/metrics
```

Output directories are create-only per episode; existing paths are refused rather than
overwritten.

---

## 7. Shutdown

CARLA was shut down cleanly after each campaign stopped. Final verification after the
second campaign: `ps -eo pid,comm | grep -i carla` returns nothing and RPC port 2000 is
free.

(A `pgrep -f CarlaUnreal` poll used during the first shutdown self-matched its own
`bash -c` command line and exited 144; the `ps`-based check above is the authoritative
one, and both agree the server is down.)

In-simulation perception sensor cleanup succeeded in **all four** collected episodes, with
all four sensors reporting `destroy_result: true` and `final_state: absent`, `cleanup_tick:
"ok"`, and zero warnings. The qualified runner's own actor cleanup also reported
`cleanup_succeeded=true` in each.

No model was retrained, no checkpoint was modified or overwritten, no other AE family was
evaluated, no quantization/ROI profile was run, no OAI was started, and no production
client or spatial-map code was edited.
