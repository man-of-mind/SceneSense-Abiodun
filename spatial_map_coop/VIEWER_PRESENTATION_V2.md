# Cooperative spatial-map viewer: presentation contract v2

The additive [replay-first UI foundation](replay_ui_v1/README.md) supplies a
read-only M3 replay launcher, capture-bound pose/clock adapter, explicit forward
observation policy, browser inspection evidence and demo artifacts. Its optional
predictor age/consistency guards preserve the defaults described below for the
existing live viewer; the live viewer is not connected or upgraded by launching
the replay.

Offline refinement, 2026-09-21, branch `spatial-map-coop-v1`.
**No live-performance and no UI-qualification claim is made anywhere in this
document.** Nothing here was validated against a running CARLA, OAI, Docker,
CUDA or network service; every statement below is either a code contract or an
offline test result.

## The three layers, and why they must not be confused

| Layer | What it is | Where it lives | May it move a coordinate? |
| --- | --- | --- | --- |
| **Measured detection** | A p025 object record from a UE's model run: `world_x/y/z`, `size_x/y/z`, `yaw_sin/cos`, `score`. | `multi_ue_v1/contracts.py`, `coop_input_v1/` | It *is* the coordinate. Never altered by anything downstream. |
| **Scientific association** | Class/time/XY/size-gated one-to-one Hungarian assignment, then freshest-source selection with **no position averaging**. | `multi_ue_v1/association.py` | No. It *selects* one member's position; it never blends. |
| **Presentation interpolation/prediction** | Causal constant-velocity display extrapolation, lane-guided display yaw, canonical footprints, colours, overlays. | `viewer_assets/*.js` | Only in pixels. Never in evidence, counters, tracks or any scientific output. |

Presentation fields are namespaced under `display.*` in
`presentation_inventory_v1.js` so a reader of a snapshot object cannot mistake a
drawn value for a measured one. The predictor returns new plain objects; the
snapshot it reads is left byte-identical (asserted by test).

## What replaced the old smoothing

The previous viewer used **lag-only** interpolation: `advanceDisplay` smoothstepped
from the *previously displayed* pose toward the *latest measured* pose over
`clamp(observed interval, 250..1600 ms)`. Because the real cadence is 100 ms but
the floor was 250 ms, the display was structurally behind the map at all times.

It is now a causal constant-velocity predictor (`presentation_predictor_v1.js`).

### Equations

For one tracked identity with the last two accepted measured samples
`(t₀, p₀, ψ₀)` and `(t₁, p₁, ψ₁)` on the **measured** capture clock:

```
dt        = (t₁ - t₀) / 1000                       seconds, 0 < dt ≤ 0.5
v_raw     = (p₁ - p₀) / dt                         m/s
v         = v_raw · min(1, v_max(class) / |v_raw|)  magnitude clamp, direction kept
ψ̇        = clamp((ψ₁ ⊖ ψ₀) / dt, ±60)              deg/s, ⊖ = shortest signed arc
```

At browser-local render time `now`, with `a` = the local time the sample arrived:

```
τ         = clamp(now - a, 0, 200) / 1000           seconds
u         = clamp((now - e) / 120, 0, 1)            e = correction start
w         = 1 - u²(3 - 2u)                          smoothstep residual weight
x_disp    = x₁ + vₓ·τ + εₓ·w
y_disp    = y₁ + v_y·τ + ε_y·w
ψ_disp    = ψ₁ + ψ̇·τ + ε_ψ·w
```

`ε` is the residual captured at the instant a measurement is adopted:
`ε = (drawn pose just before adoption) - (new measured pose)`. At `u = 0`,
`w = 1` and the drawn pose equals what was already on screen, so **a correction
never produces a visible jump**; by `u = 1`, `w = 0` and the drawn pose is pure
measurement-plus-prediction. Under a steady 100 ms cadence the residual decays
geometrically (measured in test: 1.0 m → <0.01 m within 12 updates), so
**steady-state display lag is ~zero** rather than one full update.

### Registered constants

| Constant | Value | Meaning |
| --- | --- | --- |
| `PREDICTION_HORIZON_MS` | **200 ms** | Maximum extrapolation. Past it the pose **holds**; it does not drift. |
| `CORRECTION_MS` | 120 ms | Residual decay window. Shorter than the horizon. |
| `MAX_SAMPLE_GAP_MS` | 500 ms | Two samples further apart do not define a velocity; `v = 0`. |
| `BACKWARD_RESET_MS` | 1000 ms | A measured clock jumping back this far is a restart: reset, `v = 0`. |
| `MAX_DISPLAY_SPEED_MPS` | Vehicle 25, Pedestrian 4, Cyclist 12, default 15 | Conservative class speed caps. Display guards, **not** calibrated kinematics. |
| `MAX_DISPLAY_YAW_RATE_DPS` | 60 deg/s | Yaw-rate clamp. |

### Two clocks, deliberately

Velocity magnitude uses the **measured capture clock** (`capture_timestamp_ns`,
newly exposed per stream by the service) so `m/s` is a real rate. The
extrapolation horizon uses **browser-local time since the sample arrived**,
because the browser has no synchronised relationship to the CARLA simulation
clock. The horizon is therefore an honest "how long have I been guessing" bound,
not a claim about measured age.

### Timestamp safety

* repeated measured timestamp → `REPEATED`, no velocity update, horizon keeps
  counting from the real last measurement;
* out-of-order (small backward step) → `OUT_OF_ORDER`, state preserved;
* backward jump ≥ 1 s → `RESET` with zero velocity;
* missing/`null`/`''`/`NaN` timestamp → treated as **absent**, velocity stays
  zero, change detection falls back to an opaque `sampleId` (a frame counter).
  *`Number(null)` is `0`, so absence is detected before any numeric coercion —
  otherwise a missing clock becomes a real time of zero and the predictor starts
  inventing velocities. This was a real bug caught by the offline tests.*
* non-finite position → `INVALID`, the sample is refused outright.

## Raw mode stays measurement-stepped

**Raw mode is deliberately not smoothed or predicted.** Raw mode exists to show
which measurements were actually received; interpolating it would blur exactly
that. Raw footprints therefore step when a report arrives.

Measured **ego poses and FoV wedges are predicted in both modes**, because they
are the observer's own pose rather than a detection. In raw mode this means the
camera moves smoothly while detections step — that asymmetry is intentional and
is what an honest raw view looks like.

A predicted object is never drawn as an additional detection. A single footprint
is drawn at the predicted pose, with a thin faint leader line back to the
measured point, and a **dashed** outline once the pose is held past the horizon.
The legend names both cues.

## Lane-guided display yaw, now cached

`roadHdg` previously linearly scanned **every** Town10HD road segment for
**every** non-pedestrian object on **every** animation frame — 5,874 segments, so
roughly 7M segment tests/second at 60 fps with ~20 vehicles.

It is now `RoadHeadingIndex`: a uniform grid keyed at 8 m (the acceptance
radius), queried over the 3×3 neighbourhood, allocation-free, with typed-array
storage.

**Exactness.** If some segment's closest point to the query is within 8 m, that
point lies inside the segment's bounding box and within one cell of the query;
with `cell ≥ radius` the 3×3 neighbourhood therefore necessarily contains that
segment. Ties break toward the lowest flattened segment index, which reproduces
what the original ascending-order scan with a strict `<` produced. Verified by
test over 5,100 grid queries on the real recorded Town10HD_Opt geometry
(2,639 on-road hits): **0 mismatches, 23.1× faster**.

Lane geometry influences **displayed vehicle yaw only**. No measured XY is ever
lane-snapped, and lane geometry never touches association.

## Presentation quality

* 16:9 recording stage, so framing is identical regardless of monitor size.
* Dark theme; roads drawn with a casing plus a lighter core so geometry reads on
  a recording rather than looking like a wireframe.
* Colour-blind-readable **Okabe-Ito** assignments, keeping the existing source
  semantics (UE-A = blue family `#56B4E9`, UE-B = orange family `#E69F00`,
  A+B = green family `#009E73`, pedestrian accent `#CC79A7`). *These are new
  exact values*: the previous cyan/orange/green set was not readable as a triple
  for deuteranopia. Hue is never the only cue — an A+B representation also gets a
  redundant double stroke.
* Fixed-width value columns with `tabular-nums`, so numbers do not reflow as they
  change.
* Keys: `R`/`1` raw, `F`/`2` associated, `Space` toggle, `M` metrics, `L` legend,
  `C` controls, `B` route overlay, `P` presentation mode (strips chrome).
* Cadences: snapshot poll 20 Hz, metrics panel 2 Hz, canvas at
  `requestAnimationFrame`. The road index is built once, never on a frame.

## Honest live metrics

Rules enforced in code (`coop_input_v1/live_metrics_v1.py`), not left to a
template author:

* **Every latency names both endpoints.** `edge tail-complete -> map-install`,
  `edge publish-start -> map-install`, `capture -> map-install AoI`,
  `map snapshot build (server, lock-to-JSON)`. Registering a window whose label
  contains `network latency`, `5g latency`, `radio latency`, `uplink latency`,
  `end-to-end network` or `wireless latency` **raises**.
* **Cross-clock quantities return a string, never a number.** For the HTTP demo
  transport, wall AoI is `N/A - incompatible clock domains`, because its capture
  clock is `carla_simulation_ns` and the server receipt is `unix_wall_ns` with no
  measured offset contract.
* **The HTTP demo path is not a radio.** Its POST duration may only be shown as
  `demo-control delivery time (HTTP POST, localhost)` and its inference as
  `localhost model path`.
* **Absent means N/A, never zero.** A dropped sample is dropped. A boundary
  inversion (install before tail-complete) is **excluded**, not clamped to zero.
* **Bounded, deterministic windows.** Fixed-capacity FIFO, default 120 samples,
  nearest-rank percentiles, so every reported P50/P95 is an actually observed
  sample and the same inputs always give the same output. No full-history scan.
* Direct-edge latency charts stay **empty** on a transport that carries no edge
  stage stamps, with an explicit note saying so, rather than showing zeros.

### Health endpoint correction

`/healthz` previously reported `fused_tracks` from `fusion_tracks`, the
historical greedy confidence-weighted structure. With `--multi-ue-v1` enabled
that dictionary is never populated, so a healthy conservative service reported
zero tracks. Both `/healthz` and `/api/multi_ue/v1/live_updates/healthz` now
report the actual `MultiUESpatialMapService` state: aligned/input source counts,
accepted and rejected observation counts with reasons, association count,
multi-source association count, retained vs. current-association track counts,
and the buffered source count.

These endpoints read a **cached** diagnostic recorded at the one already-existing
stateful read (`_build_multi_ue_v1_objects`). They do not call
`snapshot_latest()` themselves, because that advances the conservative track
store — a monitoring poll must not mutate scientific association state. Asserted
by test: zero snapshot calls across repeated health/metrics requests.

## Route contract

`--route-mode stage2` (default) is the historical configurable spawn-index loop
`80,85,91,94,99,80`. It reports `is_route_b = false` and
`qualification_status = STAGE2_LOOP_NOT_ROUTE_B`, and the contract **refuses to
be constructed** with a presentation label that claims Route B other than to
deny it.

`--route-mode route-b-passive` binds the canonical
`data_collection/routes/town10hd_opt_route_b_full_map_loop_v1.json`, verifies its
SHA-256 against the pin
`fc4518a8746b9417a64616b8e544f59b16b5a31b7585298a316a59662ecfd6e5` (the branch
copy and the master copy are byte-identical at this value), optionally binds the
route-progress contract, and requires two explicitly registered ego/stream
identities. `--route-overlay` then draws the canonical path as a subtle dashed
background reference.

**No Route-B run is authorized.** The contract reports
`live_route_b_run_authorized: false`, the harness refuses
`route-b-passive` combined with `--spawn-two-egos` (configurable spawn indices
cannot produce a Route-B demonstration), and the run summary lists
`ROUTE_B_LIVE_QUALIFICATION` and `COMPLETE_TWO_UE_ROUTE_B_DEMONSTRATION` under
`not_claimed`. A live qualification must be driven by the canonical route runner.

## Tests

```bash
# JavaScript: the exact assets the server inlines
node spatial_map_coop/tests/js/presentation_tests_v1.js

# Python: normalization seam, contracts, viewer endpoints
python3 -m unittest spatial_map_coop.tests.test_coop_input_v1 \
                    spatial_map_coop.tests.test_viewer_presentation_v1 \
                    spatial_map_coop.tests.test_multi_ue_v1
```

All offline. No CARLA, CUDA, OAI, Docker, socket or port; the Flask endpoints run
through the in-process WSGI test client and the town geometry comes from the
recorded fixture.
