# Depth visibility / occlusion contract — bounded CARLA smoke report

**Date:** 2026-08-27
**Terminal:** `DEPTH_VISIBILITY_SMOKE_READY_FOR_MANUAL_REVIEW` (with two caveats, §6)
**Artifact:** `data_collection/experiments/depth_visibility_smoke_v1/20260827_023930/`
**Frozen algorithm (registered before the run):** `DEPTH_VISIBILITY_ALGORITHM_V1.md`

This is a diagnostic. Nothing was integrated into the canonical v2 collector, no
dataset was collected, no model was trained or run, and no checkpoint was touched.
Final qualification is **not** claimed — the contact sheet is ready for manual review.

---

## 1. Question

Route B v2 paints every eligible pedestrian actor box as a filled mask, and CARLA
0.10 gives no usable walker semantic/instance pixels, so a fully occluded or
visually absent pedestrian stays detection+segmentation GT. Can a synchronized,
colocated depth camera separate visible / partially visible / fully occluded
pedestrian actor boxes well enough to become a GT eligibility contract?

**Answer from this smoke: yes, decisively, for occlusion by a foreground object.**
The one stage designed to test occlusion by *static scene geometry* did not
construct an occlusion and is a null result (§6.1).

## 2. What ran

One fresh CARLA server, `Town10HD_Opt`, explicit `-quality-level=Epic`, GPU
rendering on (`-RenderOffScreen`), `no_rendering_mode=False`. One stationary ego
carrying an RGB and a colocated depth camera at the exact Route B v2 mounting
(1280x720, 120 deg FOV, x=1.8 y=0.0 z=1.55 pitch=-4.0), free-running
(`sensor_tick=0.0`) in a 20 Hz synchronous world. A pool of four static actors —
one `walker.pedestrian.0015`, one `vehicle.sprinter.mercedes` occluder, two
control vehicles — with physics disabled and **teleported** between stages inside
the same world. No traffic manager, no walker AI controllers, no autopilot, no
population replenishment, no Route B loop.

The ego pose was chosen at runtime by measuring the forward corridor from the
depth camera itself (best available: 30.9 m clear at spawn index 37); no map
assumption was baked in.

## 3. Results — 7 stages x 3 synchronized frames

Per-actor means over the 3 frames. `vf` = visible fraction, `in_px` =
depth-consistent pixels at 768x432 model input resolution, `closer` = fraction of
box pixels whose measured depth is in front of the actor (an occluder).

| stage | actor | expected | dist m | area px | vf | in_px | closer | eligible @0.10 | @0.05 | @0.20 |
|---|---|---|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| S1 ped visible 10 m | walker | visible | 8.2 | 1483 | **0.673** | 384 | 0.000 | 3/3 | 3/3 | 3/3 |
| S2 ped visible 30 m | walker | visible | 28.2 | 121 | **0.626** | 39 | 0.000 | 3/3 | 3/3 | 3/3 |
| S3 ped partial behind vehicle | walker | partial | 12.8 | 690 | **0.558** | 155 | 0.113 | 3/3 | 3/3 | 3/3 |
| S4 ped heavily occluded | walker | heavy | 12.7 | 664 | **0.213** | 55 | 0.628 | 3/3 | 3/3 | 3/3 |
| S5 ped fully occluded | walker | fully occluded | 12.7 | 608 | **0.061** | 12 | 0.818 | **0/3** | 3/3 | 0/3 |
| S6 ped behind static geometry | walker | (not achieved) | 11.2 | 3221 | 0.427 | 517 | 0.000 | 3/3 | 3/3 | 3/3 |
| S7 control, visible vehicle | vehicle_a | visible | 15.1 | 5441 | 0.727 | 1481 | 0.000 | 3/3 | 3/3 | 3/3 |
| S7 control, occluded vehicle | vehicle_b | fully occluded | 12.4 | 4737 | **0.000** | 0 | **1.000** | **0/3** | 0/3 | 0/3 |

**Monotone across the constructed occlusion ladder.**
`visible_fraction`: 0.673 → 0.626 → 0.558 → 0.213 → 0.061 → 0.000.
`occluder_closer_fraction`: 0.000 → 0.000 → 0.113 → 0.628 → 0.818 → 1.000.

Separation gate: visible median 0.6503 vs fully-occluded median 0.0606 —
absolute difference **0.590** (required 0.10), ratio **10.7x** (required 5.0x).

### Sensitivity (reported only; the registered 0.10 rule was not changed)
- **0.05 would break the contract.** The fully occluded pedestrian (S5) is
  *accepted* 3/3 at 0.05. A 0.05 threshold does not reject full occlusion.
- **0.20** rejects S5 and still accepts S1–S3, but also sits directly on S4
  (0.213) with ~1 pp of margin.
- The registered **0.10** is the only tested threshold that both rejects full
  occlusion and keeps a usable margin on both sides.

### The pixel-count criterion alone is not sufficient
S5's `model_input_visible_px` is **exactly 12** — it *passes* the >=12 px
criterion and is rejected only by the `visible_fraction >= 0.10` test. If the
future contract keeps only a pixel-count floor, fully occluded pedestrians will
survive as GT. Both criteria are load-bearing.

### Heavily occluded case, reported truthfully
S4 (`vf=0.213`, `closer=0.628`, 55 px at input) is **accepted** under the frozen
rule, identically on all 3 frames.

**Correction after visual review** (`review_crops/S4_ped_heavy_occluded_review.png`):
the lateral offset I registered did not produce the intended thin lateral sliver.
The van's hood and roofline occlude the pedestrian's **lower body**, leaving the
**head and torso fully visible above the van**. So S4 is not a "small body region
visible" case at all — it is an ordinary lower-body occlusion that a detector
should be expected to find, and accepting it is the right answer rather than a
tolerated miss. The depth mask tracks the visible upper body accurately.

The consequence: **the ladder has a gap between S4 (0.213, upper body visible)
and S5 (0.061, nothing visible).** No stage in this smoke actually tests a
genuinely marginal pedestrian — a few percent of body visible through a gap. The
0.10 threshold is therefore validated as *separating clear cases*, not as
correctly placed on the hard boundary. Locating that boundary needs a follow-up
with intermediate occlusion levels.

## 4. Runtime gates

| gate | result |
|---|---|
| G1 RGB/depth identical CARLA frame id | PASS (62/62 captures) |
| G2 timestamp delta within synchronous tolerance | PASS — **max delta 0.0 s exactly** |
| G3 no missing / duplicate / out-of-order frames | PASS (consecutive ticks within every stage) |
| G4 RGB and depth non-empty | PASS |
| G5 depth finite and physically plausible | PASS |
| G6 visible materially above fully occluded | PASS (0.590 abs, 10.7x) |
| G7 fully occluded rejected | PASS |
| G8 clearly visible accepted | PASS |
| G9 partially visible accepted | PASS |
| G10 actor/sensor cleanup succeeded | reported FAIL in-run — **accounting bug, see §6.2** |

Projection was verified independently against the retained frames, not by eye:
S1's box `(631,301)-(649,387)` has an in-box median depth of **8.18 m** against
an actor interval of 7.94–8.45 m, and the crop shows the box tightly framing a
real pedestrian. S5's box carries the same 12.43–12.94 m actor interval but an
in-box median of **9.56 m** — the van in front. Geometry and depth agree.

## 5. Prior-art finding: `rasterize_person_regions_depth` must not be reused

`pole_lraspp_multimodal_fusion/pole_lraspp_multimodal_fusion/common.py:320` is
**not** a visibility contract and should not be adopted as one:
- it tests against a single scalar actor distance with asymmetric pads
  (-1.5 m / +1.0 m) instead of the actor's own near/far interval;
- **line 350 restores the full ellipse whenever depth carving keeps under 12% of
  the region.** A fully occluded pedestrian is therefore silently repainted as
  fully visible — precisely the failure this study exists to remove. On S5 the
  measured `vf` is 0.061, i.e. below that 12% fallback trigger.

This smoke implements the interval test in `DEPTH_VISIBILITY_ALGORITHM_V1.md` §3.

## 6. Caveats — read before manual review

### 6.1 S6 is a null result, not a pass
The deterministic placement for "hidden by static scene geometry" probed the
depth image on a row 40 px **below** the principal point, which hits the **road
surface**, not a facade. Placing the walker 4 m beyond a ground hit put it in the
open on a sidewalk (`closer = 0.000`, `vf = 0.427`, visibly unoccluded in the
contact sheet). **Occlusion by static scene geometry was therefore not tested.**
No gate depended on S6. It must be redone with a horizon/above-horizon probe
before the contract can claim static-geometry coverage.

### 6.2 G10 is an accounting bug, and cleanup did succeed
All 7 actors (ego, 2 sensors, 4 pool actors) returned `destroy() == True`. The
in-run residual scan then read `world.get_actors()` before the destruction batch
was committed on the next tick, so it reported 4 phantom survivors and drove the
script's own terminal to `RUNTIME_FAILED`. Verified independently against the
live server after the run: **0 vehicles, 0 walkers, 0 sensors**. A one-line fix
(`wait_for_tick` before the scan) was applied to the script **after** the
`20260827_023930` artifacts were produced and is marked as such in the source;
the smoke was **not** re-run to flip the boolean.

### 6.3 Other scope limits
- Single walker blueprint (`walker.pedestrian.0015`), single occluder
  (`vehicle.sprinter.mercedes`), single ego pose, one weather/time of day.
- Stationary ego and static actors. Motion blur, rolling-shutter and
  ego-motion effects are untested.
- The visible fraction is measured over the projected **box**, so ground pixels
  at the actor's own depth leak in near the feet. This inflates `vf` slightly
  and is the likely source of S5's residual 0.061 rather than true visibility.
- `DEPTH_TOLERANCE_M = 0.25` was fixed in advance and never swept.

## 7. Verdict

Synchronized depth **can** provide a trustworthy pedestrian visibility/occlusion
contract for Route B, for occlusion by foreground objects, at the registered
`visible_fraction >= 0.10` + `>= 12 px at model input` rule. The vehicle control
pair behaves exactly as the semantic path would (visible 0.727 accept, occluded
0.000 reject), which is independent confirmation that the depth test is measuring
occlusion and not an artifact of the walker rendering gap.

Two items must close before this becomes a collection contract: redo S6 with a
correct static-geometry probe, and confirm the rule under ego motion.

## 8. Provenance

| item | value |
|---|---|
| CARLA launch | `./CarlaUnreal.sh -RenderOffScreen -nosound -quality-level=Epic -carla-rpc-port=2000 -carla-server -benchmark -fps=20` |
| server / client version | 0.10.0 / 0.10.0 |
| map | `Carla/Maps/Town10HD_Opt` |
| client wall time | 16.05 s |
| simulated time | 16.75 s (335 world ticks at 0.05 s; 62 captured frames) |
| client peak RSS | 585,612 KiB (572 MiB) |
| GPU peak (whole device) | 6,928 MiB during the run |
| GPU after shutdown | 1,071 MiB — identical to the pre-run baseline, context fully released |
| CARLA shutdown | `SIGTERM` did **not** exit within 48 s; `SIGKILL` was required. GPU memory was released either way. |
| actor cleanup | 7/7 destroyed; live world afterwards had 0 vehicles / 0 walkers / 0 sensors |

### Files created
- `data_collection/route_b_depth_visibility/DEPTH_VISIBILITY_ALGORITHM_V1.md` (frozen before the run)
- `data_collection/route_b_depth_visibility/carla_depth_visibility_contract_smoke_v1.py`
- `data_collection/route_b_depth_visibility/DEPTH_VISIBILITY_SMOKE_REPORT.md` (this file)
- `data_collection/experiments/depth_visibility_smoke_v1/20260827_023602/` (aborted first invocation, §9)
- `data_collection/experiments/depth_visibility_smoke_v1/20260827_023930/` (8.2 MB): `resolved_config.json`,
  `per_frame_visibility_metrics.csv`, `summary.json`, `stage_manifest.json`,
  `frame_alignment_evidence.json`, `contact_sheet.png`, `provenance_frames/` (one
  RGB jpg + one float16 depth npz per stage), `review_crops/` (zoomed annotated
  S1/S2/S4 panels, generated offline from the retained frames after the run)

No existing file was modified. The canonical v2 collector was not touched.

## 9. Invocation history — stated plainly

Two client invocations were made against CARLA, not one.

1. `20260827_023602` — died after **0.26 s** inside `connect()`, before any world
   interaction, with `RuntimeError: std::exception`. Cause: `world.get_settings()`
   raises `bad_optional_access` on a freshly booted server that has never had
   `load_world` called; `run_route_b_density_loop.py:1046` does
   `client.load_world(...)` first. Zero world ticks, zero frames, zero
   measurements — no evidence of any kind was produced.
2. `20260827_023930` — the reported run.

Between them, three client-side defects were fixed and two read-only probes were
run against the server: the `load_world` bug above; wrong blueprint ids (this
build has `vehicle.sprinter.mercedes`, `vehicle.lincoln.mkz`,
`vehicle.mini.cooper` — `vehicle.tesla.model3` and `walker.pedestrian.0001` do
not exist); and the addition of the forward-corridor ego pose selection. **No
threshold, tolerance or gate was changed** — the registered rule in
`DEPTH_VISIBILITY_ALGORITHM_V1.md` is exactly as frozen before the first launch.
A separate shell-quoting error caused one launch attempt to fail before the
Python process started; it never reached CARLA.
