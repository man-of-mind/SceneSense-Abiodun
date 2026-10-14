# Replay spatial map: heading and presentation repair v2

> **Current default (v3): [ego traffic view with tiered display tracks](README_V3.md).** `/` now serves v3; this v2 UI remains at `/v2`, and `/compare-v3` shows both. The section below documents v2.

From the repository root, one command starts the read-only UI and opens a browser:

```bash
python3 -m spatial_map_coop.replay_ui_v1.server --open
```

[Open the replay](http://127.0.0.1:8765), or [compare both versions](http://127.0.0.1:8765/compare).
`--port` changes the port; `--capture` points to a relocated copy of the registered
V10 capture. Without a graphical browser, omit `--open` and use the printed URL.
No CARLA, OAI, inference, training or live ingest is started. The application has
only read endpoints. Existing NumPy, SciPy, Pillow and Flask are sufficient.

## Demonstration and verification

- [Normal-speed side-by-side video](artifacts_v2/before-after-1x.webm): departure,
  turn, dense/stopped scene and interrupted stream, identical source, 45 m radius,
  heading-up, 1× speed and current/coast selections.
- Screenshots: [departure](artifacts_v2/01-departure-before-after.png),
  [turn and building shadow](artifacts_v2/02-turn-before-after.png),
  [dense stopped scene](artifacts_v2/03-dense-before-after.png),
  [expiry](artifacts_v2/04-interrupted-before-after.png),
  [improved UI](artifacts_v2/05-improved-ui.png),
  [raw provenance inspection](artifacts_v2/06-raw-provenance.png).
- [Browser report](artifacts_v2/BROWSER_CHECK.json),
  [identical-inventory motion comparison and cluster members](artifacts_v2/MOTION_AND_CLUSTERS.json),
  [synthetic truth fixtures](artifacts_v2/FILTER_FIXTURES.json),
  [heading lineage, exact joins, source hashes and geometry validation](artifacts_v2/HEADING_GEOMETRY_EVIDENCE.json).

The original [report](README_V1_HISTORICAL.md), `artifacts/` evidence, and
[baseline implementation](baseline_v1/) are preserved. Baseline source hashes
match the first task's saved validation. `/before` serves those assets with the
same raw replay data and their original unqualified-heading behavior.

## Corrected heading path

`heading_ingest_v2.py` is an explicit opt-in installation revision. It reuses the
actual frozen baseline normalizer and locked map insertion, changes only heading
fields before insertion, and has no runtime launcher. **Production, pinned wire
protocols, historical map records and the moving-ego server were not edited.**

The producer trace establishes world yaw, not camera yaw:

1. Dynamic targets use `transform.rotation.yaw` from
   `carla_collect_parked_ego_fusion_training_data.py`. Static positives use world
   `bbox_rotation_yaw_deg` via `route_b_v3_1_clean_base_v1/build_contract_v1.py`.
2. FCOS `data.py` forms sine/cosine targets directly from `object_yaw_deg`.
   `size_x` is length along object-local +X; `size_y` is width along +Y.
3. FCOS `model.py` rotates local XYZ by the camera extrinsic, but only normalizes
   the yaw pair. `infer.py` copies the pair and dimensions into prediction rows.
   CARLA local coordinates are forward/right/up; world yaw rotates +X toward +Y.
4. Therefore the explicit `p025_world_yaw.v1` contract uses
   `wrap(degrees(atan2(yaw_sin, yaw_cos)))`, **without camera rotation or +90°**.
   This qualifies the representation, not the model's orientation accuracy.

The adapter keeps legacy packets byte-equivalent when no new contract is selected.
Zero/near-zero pairs, missing/nonfinite values, unsupported contracts, and
conflicting degree/pair representations yield `valid=false`, a reason, and null
yaw. It never substitutes zero as a qualified heading. Cardinal directions,
wraparound, normalization, score-filtered indices, and all 132 retained records
from frames 1480/2020/2748 pass through the actual installation function in an
isolated offline test process. Removing the heading fields produces identical
coordinates, dimensions, scores, classes, counts and object IDs to v1.

Replay corrections are separate from `raw`. Each requires an exact
run/cell/stream/frame/capture-ns binding, installed ordinal, unique upstream
prediction/candidate identity, and exact XYZ/dimensions/score/class agreement.
Upstream prediction indices can have gaps after producer filtering; they are not
mistaken for installed ordinals. Provenance includes file and record SHA-256.
Missing joins leave the current heading unqualified; a previously supported
heading may persist only as a labelled private display estimate. Roughly half of
these replay frames have retained upstream predictions; none are reconstructed
by nearest-neighbor heading assignment or by GT actor identity.

**Integration status:** ready as a tested, opt-in adapter for a separate live
integration change. It is not deployed or a drop-in legacy renderer contract:
that change must select the producer contract explicitly, preserve invalid/null
headings and their qualification, and prevent downstream +90°/road overrides.
Live producer-version binding, concurrency/service integration and live clock
mapping remain separate work. No scientific accuracy qualification is claimed.

## Building approximation and object policy

The calibrated 120° camera FoV remains a faint dashed outline. The fill is a
separate first-boundary line-of-sight polygon cast from the mounted camera origin.
Corner rays include both sides of every relevant vertex; a 20 m segment grid
indexes intersections and the last exact polygon is cached. Shadows behind
buildings are excluded, as well as interiors. Rendering keeps the original town
backdrop and calibrated pose binding.

The stored map names `Carla/Maps/Town10HD_Opt`; the capture registration names the
same Town10 route. The existing extractor creates world-metre XY projections of
environment-object bounding boxes, filtering height <2 m, area <20 m², volume
<80 m³ and road distance >20 m. There are 458 retained footprints. Independent
capture-pose-to-lane distance P95 is 1.16 m without fitting any coordinate transform.
The asset revision is not retained: this is consistent town geometry, not proof
of identical facade meshes. Missing/invalid geometry disables the estimate.

Default building withholding requires **one boundary segment to cover the entire
expanded convex object footprint**, with 0.3 m footprint expansion, 0.5 m depth
clearance and 0.25 m endpoint clearance. Unknown heading uses a circumscribed box.
Partially blocked, corner-grazing, intersecting or invalid footprints remain drawn
with `VISIBILITY_UNCERTAIN`. Completely covered objects get `STATIC_MAP_OCCLUDED`.
The checkbox exposes them again; raw/debug always exposes the original records.

This is strictly a **2D static-building approximation**. Dynamic vehicles, heights,
transparent surfaces, bridges, omitted buildings and detector geometry errors
remain unresolved. `NO_MODELED_OBSTRUCTION` means only that no modeled boundary
blocked the footprint. The separate qualified visibility state stays unknown;
strict-visible mode still withholds unqualified objects. Clipping never changes
association, scientific observations, GT, metrics or alerts.

## Dense candidates and conservative consolidation

The inspected frames contain 38, 40 and 54 **same-frame** candidates. Current mode
cannot show prior-frame remnants; optional coasting explicitly identifies those.
Backend association fragmentation and swaps can change presentation continuity,
but cannot explain multiple candidates already present in a single frame.

Examples are deliberately distinguished:

- 1480 includes a clear foreground bus/traffic and many alternative model boxes.
  Candidates 14/31 meet same-frame overlap tests but lack the required prior-frame
  support, so neither is removed at the inspected event.
- 2020's candidates 0/20/29 overlap the foreground vehicle in RGB yet have differing
  estimated geometry; close centroids alone do not consolidate them. The
  pedestrian and nearby vehicles remain separate classes/hypotheses.
- 2748 has 109 pairs within 1 m. Some project to entirely different image regions:
  for example candidates 6 and 14 are nearly coincident in world XY but their
  image boxes occupy different vertical regions. These remain unresolved
  localization/candidate conflicts, drawn as amber diamonds instead of convincing
  vehicle outlines. At the inspected event, only 6/43 also have sufficient temporal
  support to become one labelled display representative.

A group requires same source/frame/class; image IoU ≥0.45 and containment ≥0.85;
center distance ≤half the smaller predicted width; all dimension ratios ≥0.65;
qualified yaw agreement ≤25°; and a compatible duplicate pair in the most recent
qualified frame within 500 ms (distinct prior candidates, box IoU >0.25 and world
movement <3 m for both). All members must agree pairwise, preventing chained
merges. These gates require image, footprint and temporal support; they do not
prove a physical identity. Missing current image evidence prevents consolidation.

The highest score wins, then lexical record ID breaks ties. Its measured position
is retained exactly; group centers are never averaged. `×N` badges, raw counts,
`CONSOLIDATED_HYPOTHESIS` rows, member candidate metadata and provenance expose every
member. The grouping checkbox restores individual hypotheses. Diamonds represent
conflicting image/world evidence, not a new class or scientific error label.
Most crowded candidates remain inspectable rather than being silently discarded.

## Causal motion and measured trade-offs

The responsible display layers were noisy centroid differencing, association
changes, alternating zero/large velocity, heading fallback changes, and correction
resets. The private replay projection now uses deterministic global one-to-one
matching against capture-time predicted positions, last measurements, class,
size, image overlap and available heading. The original backend track ID remains
in each inspection record. No scientific track store is modified.

| Setting | Current behavior |
|---|---|
| View/source | Selected source, current forward observations, heading-up, 45 m radius; unchanged freshness and frustum checks |
| Time | Causal arrivals only; ego and objects predict to the same replay time; overview shows only received route poses |
| Input buffering | **0 ms**; no future observations enter tracking/filtering |
| Filter | Alpha-beta at capture time: objects α=0.4, β=0.1; ego α=1, β=0.5 |
| Display transition | 100 ms cubic Hermite continuity bridge, including velocity continuity, targeting the age-compensated state |
| Bounds | 500 ms total capture-age prediction; 350 ms current freshness; 500–750 ms coast fade; expiry at 750 ms |
| Matching | Minimum of predicted/last-position distance ≤3 m; size ratios ≥0.5; global cost = 0.65 predicted distance + 0.35 last distance + 0.4 image disagreement + 0.3 heading disagreement; unmatched cost 5 |
| Stops | Raw measured speed <0.25 m/s zeros extrapolation; noisy stops retain filter uncertainty |
| Reacquisition | Up to 750 ms, zero velocity after >500 ms gap; preserve ID and bridge only within 3 m; otherwise reset |
| Guards | Innovation >5 m resets; original class speed caps (vehicle 25, pedestrian 4 m/s); object yaw changes ≤90°/s; no road-heading override |
| Memory | 256 tracks/events, 8 sources, 8192 correction samples, bounded clip/image caches |

The deterministic 60 Hz comparison below disables **both static suppression and
consolidation**, keeping the full inventory and the same view settings. Changes
in mean coverage are caused by display-position region checks, not a score gate.

| 18 s sequence + expiry | P95 frame step, before → after (m) | P95 correction residual (m) | Display resets | New display IDs | Mean shown |
|---|---:|---:|---:|---:|---:|
| Departure | 0.448 → 0.305 | 2.616 → 1.460 | 107 → 11 | 1012 → 956 | 21.30 → 21.40 |
| Turn | 0.434 → 0.283 | 2.589 → 1.432 | 119 → 7 | 894 → 861 | 25.22 → 25.57 |
| Stopped ego | 0.341 → 0.184 | 2.206 → 1.111 | 56 → 1 | 573 → 578 | 20.31 → 20.13 |
| Avenue | 0.456 → 0.323 | 2.672 → 1.522 | 132 → 14 | 1188 → 1129 | 18.33 → 18.51 |

P95 second differences also fall, but are **not** called pure jitter: real
acceleration and turns contribute. Backend/display identity ambiguity persists;
the stopped sequence creates five more display IDs. Maximum individual steps are
not uniformly reduced (departure 0.743 → 0.770 m). These limits are disclosed in
`MOTION_AND_CLUSTERS.json`, including all-inventory and consolidation variants.

Synthetic fixtures with known truth distinguish motion from noise. For a
stationary point with deterministic 0.18 m measurement noise, displayed RMSE falls
0.393 → 0.096 m; at a genuine 4 m/s, RMSE falls 0.728 → 0.097 m without freezing
motion, and mean signed lag after warmup is approximately −0.16 ms. A 5 m/s turn
retains 0.535 m RMSE. Noisy abrupt stopping settles to the fixture's 0.15 m / 0.5 m/s
criteria after **1.17 s**, versus 0.65 s before, including the existing 200 ms
capture-to-arrival age. This is a real filter trade-off, not an added input buffer
or a claim of zero latency. Clean measured stops zero velocity immediately, then
finish the continuity bridge. Real-data capture-time filter offsets and residuals
are reported separately from model accuracy.

The actual browser video uses simultaneous 2880×1240 recording, so occasional
missed frames enlarge per-frame steps on both sides; its paired clocks differ by
less than a frame. The standalone 1440×1100 warmed run is measured independently
in `BROWSER_CHECK.json`: **60.00 fps**, frame P95
**16.7 ms**, drawing P95 **2.7 ms**.
Side-by-side recording frame P95 is 33.4 ms, identical on both sides.
Browser coverage (mean per sampled frame) is disclosed below; raw counts match
on both sides, and uncertainty diamonds remain drawn.

| Browser segment | Before shown | After shown | Consolidated | Building withheld | Uncertainty diamonds |
|---|---:|---:|---:|---:|---:|
| departure | 21.92 | 22.07 | 0.00 | 0.00 | 0.81 |
| turn | 30.60 | 30.42 | 0.23 | 0.21 | 0.31 |
| stop | 19.89 | 19.56 | 0.95 | 0.00 | 1.61 |
| interruption | 11.40 | 11.89 | 0.22 | 0.00 | 0.23 |

The no-message tail fades and expires; strict visibility, raw audit, repeatable
seek, source bounds and a narrow viewport pass. No external request occurs.

## Reproduce focused checks

```bash
node spatial_map_coop/tests/js/replay_refinements_v2.js
node spatial_map_coop/tests/js/replay_tests_v1.js
node spatial_map_coop/tests/js/presentation_tests_v1.js
python3 -m unittest spatial_map_coop.tests.test_replay_heading_v2 \
  spatial_map_coop.tests.test_replay_ui_v1
python3 -m spatial_map_coop.replay_ui_v1.evidence_v2
node spatial_map_coop/replay_ui_v1/filter_fixtures_v2.cjs
```

The new tests cover actual heading installation, legacy equivalence, missing map,
shadow/corners/partial footprints, adjacent/parked/crossing tracks, exact joins,
raw immutability, wraparound, stops, turns, reacquisition, out-of-order events and
expiry. There are 14 new JS scenarios, 23 existing replay scenarios, 105 existing
presentation checks and 21 Python checks. Unaffected viewer/adapter checks passed
in the preserved v1 validation; no unrelated experiment suite was rerun.

With the server running, development-only Playwright records the comparison:

```bash
PLAYWRIGHT_MODULE=/path/to/playwright \
PLAYWRIGHT_BROWSERS_PATH=/path/to/playwright-browsers \
node spatial_map_coop/replay_ui_v1/browser_compare_v2.cjs
```

The saved run uses the pre-existing Playwright module and a browser runtime in
`/tmp/scenesense-replay-browser`; these are tooling, not application dependencies.
`compare_metrics_v2.cjs` reads bounded JSON clips `/tmp/replay-{departure,turn,stop,avenue}.json`;
export them with `python3 -m spatial_map_coop.replay_ui_v1.evidence_v2 --clips-dir /tmp`
to repeat the deterministic comparison.
No capture, historical measurement, GT/scorer, alert or pinned production source
was changed. No commit, push, deletion or live integration was performed.
