# Cooperative Occlusion Reasoning in `manual_control_ar_v13.py`

**Implementation status:** deterministic v13 demonstration baseline, 2026-08-31
**Primary files:** `manual_control_ar_v13.py`, `cooperative_occlusion_v1.py`
**Purpose:** explain how the ego blind area, cooperative occlusion removal, occluded-object state, spatial-map rendering, and driver-warning authorization are calculated.

## 1. Scope and interpretation

The v13 implementation demonstrates the geometry and user-interface behavior of cooperative perception. It uses CARLA's ground-truth actor transforms and bounding boxes together with **virtual**, co-located RGB-camera/radar metadata. The virtual pairs are not CARLA sensor actors: they are not spawned, started, subscribed, or streamed.

The current model is therefore a deterministic 2-D reference implementation for answering three questions:

1. Which part of the ground plane is inside the ego sensor pair's field of view but hidden by an opaque footprint?
2. Which of those hidden areas can be observed from another active sensor site?
3. Is a role-labelled rogue pedestrian or rogue vehicle visible to the ego pair, visible only to a peer pair, or still unresolved?

It is not yet a measured RGB/radar perception pipeline, learned multi-modal fusion result, 3-D visibility model, or validated safety function. Those boundaries are detailed in [Section 15](#15-current-limitations-and-non-claims).

## 2. End-to-end calculation

At each top-down-map refresh, v13 performs the following data flow:

```text
CARLA actor registry + cached Town10 building footprints
                         |
                         v
Build virtual camera/radar sites on ego, vehicles, walkers,
and configured traffic-light roots
                         |
                         v
Keep sites in the ego-forward inventory region and select the
requested number of active pairs (or radar-only sites in a
network-degraded zone)
                         |
             +-----------+-----------+
             |                       |
             v                       v
Exact target-footprint          Approximate raster
visibility tests                visibility masks
             |                       |
             v                       v
Per-target ego/peer state       Remaining ego-blind area
             |                       |
             +-----------+-----------+
                         v
Spatial-map actor/label rendering and fresh visibility snapshot
                         |
                         v
Ego-camera red box, PEDESTRIAN label, and proximity warning gate
```

The exact target test and raster map layer share sensor and occluder inputs, but they deliberately use different calculations:

- **Object visibility is authoritative for object state and warnings.** It traces exact line segments to deterministic points across the target footprint.
- **The colored occlusion zone is a display approximation.** It rasterizes angular shadow polygons at reduced resolution for a responsive 10 Hz map overlay.

Small disagreements at shadow boundaries are therefore expected and do not indicate that two contradictory object-state decisions were fused.

## 3. Terminology

| Term | Meaning in v13 |
|---|---|
| Sensor site | One host location with a co-located virtual camera marker and radar marker. |
| Active sensor pair | A selected site where both virtual modalities participate in normal-mode geometry. The sensors still collect no data. |
| Sensor-site inventory region | The ego-relative area in which sites are eligible for display and selection. Default: 40 m and +/-45 degrees in front of the ego. |
| Sensor FoV | The horizontal angular cone and range used to test what one selected camera or radar could see. |
| Occluder | A 2-D opaque building, vehicle, or pedestrian footprint that can block a sight line. |
| Ego blind area | Area inside the union of active ego FoVs but inside an ego sensor's modeled shadow. |
| Recovered area | Part of the ego blind area covered by an unshadowed peer FoV. |
| Remaining occlusion | Part of the ego blind area that no active peer can see. This is the purple map layer. |
| Cooperative reveal | A target visible to at least one peer sensor but not visible to an ego sensor. |
| Network-degradation zone | A cellular-network policy region that changes active modality selection and displayed metrics. It is not a geometric occlusion zone. |

Two ranges must not be confused:

- `--spatial-map-sensor-forward-range` and `--spatial-map-sensor-forward-half-angle` decide **which host sites may be selected**.
- `--cooperative-*-horizontal-fov` and `--cooperative-*-range` decide **what each selected modality can see**.

Increasing an individual sensor's FoV does not make a host outside the sensor-site inventory region eligible.

## 4. Coordinate and footprint model

All cooperative visibility calculations use CARLA world `(x, y)` coordinates in metres. Height, pitch, roll, vertical FoV, and elevation differences are ignored.

### 4.1 Actor footprints

For a vehicle or pedestrian, `get_actor_footprint_points()` transforms the four corners of the CARLA bounding box into world coordinates. The calculation includes:

- bounding-box local location and rotation;
- actor transform and yaw; and
- the bounding-box half-length and half-width.

The resulting oriented quadrilateral is used both to draw the actor on the map and to represent it as an opaque ground-plane occluder.

### 4.2 Building footprints

Building footprints are cached when the top-down renderer initializes. v13 obtains CARLA environment objects labelled `Buildings`, converts their bounding boxes into rectangular ground footprints, and retains likely road-adjacent structures using these filters:

- minimum height: 2 m;
- minimum footprint area: 20 m²;
- minimum volume: 80 m³;
- at least one footprint-edge sample within 20 m of a sampled driving road; and
- edge sampling interval: 5 m.

At runtime, only cached building footprints whose bounds intersect the current ego-centred map square are supplied to the visibility calculation.

### 4.3 Dynamic occluders and targets

Every map-visible vehicle and pedestrian contributes its current footprint as an occluder, including the ego vehicle. For an individual sight-line test, the current sensor host and current target actor are excluded by actor ID, so neither can occlude itself.

Only strictly role-labelled `spawn_blocker_v5.py` hazards are currently evaluated as gated cooperative targets:

- pedestrian role: `pedestrian_blocker_v5_<positive index>`;
- reactive vehicle role: `reactive_blocker_v5_<positive index>`.

Ordinary traffic is still displayed as scene context from CARLA ground truth; it is not withheld by this target-state gate.

## 5. Virtual sensor-pair inventory and selection

### 5.1 Site creation

v13 creates virtual co-located camera and radar markers for:

- the ego vehicle;
- every live non-ego vehicle;
- every live pedestrian; and
- configured traffic-light roots, default IDs `14,24,11`.

For actor-hosted sites, both modality markers share a forward mount derived from the actor bounding box and share the actor yaw. The longitudinal mount is the bounding-box centre plus its forward half-length and a 0.05 m margin. Pedestrian mount height is clamped to 1.45-2.00 m; vehicle mount height is at least 0.55 m. These heights affect marker metadata, not 2-D visibility.

Traffic-light markers use the live traffic-light root when available and orient at the root yaw plus 90 degrees. A catalog fallback uses yaw zero. Their displayed height is 15 m above the root; again, only `(x, y, yaw)` affects the current visibility model.

The markers contain metadata and synthetic IDs only. They do not reference a live sensor actor and are not stream eligible.

### 5.2 Eligibility region

Before activation, a site's mount must be:

1. inside the current top-down-map square; and
2. inside the ego-forward inventory sector.

With ego planar position `E`, normalized forward direction `f`, and site position `S`, define:

```text
v = S - E
d = ||v||
theta = acos(clamp((v dot f) / d, -1, 1))
```

The site is eligible when `d` is no greater than the configured forward range and `theta` is no greater than the configured forward half-angle. A numerically co-located site is accepted without calculating a bearing. The defaults are 40 m and 45 degrees. Side and rear sites are not scanned or shown.

### 5.3 Normal-mode activation

Outside network-degraded zones:

1. Mark every eligible site inactive.
2. Keep only sites containing both a camera and a radar marker.
3. Sort sites by planar distance from the ego.
4. Pin the ego pair first whenever the requested count is positive.
5. Select the nearest `N` complete sites.
6. Mark both camera and radar active at each selected site.

The active-site count includes the ego site. Thus a budget of five normally means one ego pair plus four peer pairs, subject to site availability.

Selection uses a 2 m distance hysteresis: a previously selected moving site is retained until it lies more than 2 m beyond the current distance cutoff. This reduces marker flicker and rapid site swapping.

The target's own selected pair may consume a site slot, but it is excluded from detecting its host target.

### 5.4 Degraded-zone activation

When network degradation is active, every camera marker remains inactive and the nearest configured number of radar-capable sites is selected. Only those radar markers contribute to FoV coverage and object visibility. This is a selection policy only; no radar sensor is started.

## 6. Single-sensor FoV calculation

For a sensor at `S = (sx, sy)`, yaw `psi`, and target probe `P = (px, py)`:

```text
dx = px - sx
dy = py - sy
d = sqrt(dx^2 + dy^2)
beta = atan2(dy, dx) in degrees
alpha = wrap_to_[-180,180)(beta - psi)
```

For horizontal FoV `phi` and range `R`, the probe is inside the sensor FoV when:

```text
d <= R
and
abs(alpha) <= phi / 2
```

The implementation uses an epsilon of `1e-7`, so range and angle boundaries are inclusive within numerical tolerance. Valid configured horizontal FoV is greater than zero and no greater than 180 degrees; range must be positive.

The default modality limits are:

| Modality | Horizontal FoV | Range |
|---|---:|---:|
| Virtual RGB camera | 90 degrees | 60 m |
| Virtual radar | 35 degrees | 80 m |

`--cooperative-sensor-pair-horizontal-fov` replaces both horizontal FoVs with the same value. It cannot be combined with either modality-specific FoV option.

For map display, the bounded FoV is represented by a fan polygon from the sensor origin through 25 arc samples (24 arc segments) between `psi - phi/2` and `psi + phi/2` at distance `R`.

## 7. Exact per-object visibility

The object-state calculation is designed to answer whether **any exposed part of the target footprint** is visible, rather than testing only its centre.

### 7.1 Target probes

`target_visibility_points_xy()` creates deterministic probe points in this order:

1. the actor's reported `(x, y)` centre;
2. the footprint-vertex centroid;
3. four evenly spaced samples on every footprint edge; and
4. a point halfway from the actor centre to every boundary sample.

For edge from vertex `vi` to `vi+1`, with four subdivisions:

```text
q(i,k) = vi + (k / 4) * (vi+1 - vi),  k = 0,1,2,3
m(i,k) = 0.5 * (centre + q(i,k))
```

Duplicate points are removed within the geometry epsilon. A normal centred quadrilateral therefore has 33 unique probes: one centre, 16 boundary probes, and 16 halfway probes. It can have 34 if the actor origin and footprint centroid differ. If the footprint is absent or invalid, the calculation safely falls back to the actor centre.

These finite samples make partial exposure visible in the common cases where the target centre is hidden but a side is exposed.

### 7.2 Line-of-sight test

For each probe that is inside the sensor's FoV and range, `segment_blocked_before_target()` traces the 2-D segment from the sensor to the probe.

For each opaque footprint:

1. Skip it if its actor ID is the sensor host or target.
2. Reject it quickly if its axis-aligned bounds do not intersect the sight-line bounds.
3. Treat the line as blocked if the sensor or target probe lies inside/on the opaque polygon.
4. Otherwise test the line against every polygon edge.
5. Treat a polygon-edge intersection strictly before the target as blocked.

Tangencies and collinear overlap are treated as opaque intersections. Endpoint tolerance prevents the sensor origin or target endpoint from being counted as an intermediate obstruction.

### 7.3 Visibility rule

For sensor `s`, target `T`, target probes `Q(T)`, FoV predicate `F_s(q)`, and clear-line predicate `L_s(q)`:

```text
visible(s, T) = exists q in Q(T): F_s(q) and L_s(q)
```

In plain language, one clear in-FoV footprint probe is enough for that sensor to detect the target. The loop stops immediately after the first clear probe to keep the 10 Hz display path responsive.

For an observation:

- `inside_fov` is true if at least one evaluated probe is inside the FoV;
- `visible` is true if at least one such probe has clear line of sight; and
- `blocked` is true if probes enter the FoV but none is visible.

This is the important v13 correction to centre-ray-only behavior. A blocked centre does not hide the whole actor when another sampled part of the footprint is exposed.

## 8. Occlusion-zone map calculation

The colored map layer describes areas, not individual actors. It uses approximate shadow polygons and raster masks for performance.

### 8.1 Shadow cast by one occluder

For one sensor and one opaque polygon, `occlusion_shadow_polygon_xy()`:

1. computes the relative bearing of every occluder vertex;
2. finds the minimum sensor-to-edge distance;
3. intersects the polygon's angular span with the sensor's horizontal FoV;
4. handles FoV-boundary crossings and the +/-180-degree wrap; and
5. extrudes the intersected angular interval from the near distance to the sensor range.

The result is a quadrilateral with the conceptual vertices:

```text
left bearing at near distance
left bearing at maximum range
right bearing at maximum range
right bearing at near distance
```

The near distance is at least 0.05 m. An occluder outside sensor range produces no shadow. If the sensor lies inside the occluder polygon, the complete sensor FoV is treated as shadowed.

This angular extrusion is intentionally conservative and approximate; it is not the line-segment test used for target state.

### 8.2 Per-sensor visible mask

For active sensor `i`, let:

- `F_i` be its rasterized FoV fan;
- `H_i,j` be the approximate shadow cast by occluder `j`; and
- `H_i` be the union of those shadows, excluding the sensor host.

Then:

```text
H_i = union_j(H_i,j)
V_i = F_i intersection not(H_i)
```

`V_i` is that sensor's modeled unshadowed ground coverage.

### 8.3 Ego and peer mask fusion

The active ego camera/radar masks and all active peer masks are combined as follows:

```text
E_raw     = union of F_i for active ego modalities
E_visible = union of V_i for active ego modalities
P_visible = union of V_i for all active non-ego modalities

E_blind   = E_raw intersection not(E_visible)
Recovered = E_blind intersection P_visible
Remaining = E_blind intersection not(P_visible)
```

Interpretation:

- `E_blind` is what the ego's active virtual modalities should cover angularly but cannot see because of modeled shadows.
- `Recovered` is the part of that ego blind area covered by at least one clear peer view.
- `Remaining` is the part still hidden from every active peer. This is drawn as the purple remaining-occlusion layer.

The displayed remaining percentage is:

```text
100 * pixel_count(Remaining) / pixel_count(E_blind)
```

It is `N/A` when `E_blind` has no pixels. The denominator is the ego blind area, not the whole spatial map and not the entire world outside the ego FoV.

For performance, v13 builds these masks at approximately one-quarter map width and height, with a minimum of 32 by 32 pixels, then expands them using nearest-neighbor interpolation. The camera and radar FoV tints are unions of the corresponding active modality fans.

An object outside every ego FoV does not lie in `E_blind`; nevertheless, a peer can still detect it and classify it as a cooperative reveal. The object-state model is intentionally broader than the purple in-ego-FoV blind-area display.

## 9. How multiple sensor pairs remove occlusion

Every active peer adds an independently located and oriented coverage mask and an independent set of target sight lines. There is no majority vote. Fusion is existential OR fusion:

```text
visible_to_ego(T) = any active ego modality sees T
visible_to_peer(T) = any active non-ego modality sees T
network_detected(T) = visible_to_ego(T) or visible_to_peer(T)
```

For the area overlay, peer views contribute to the union `P_visible`. A newly activated pair reduces the remaining occlusion only where its unshadowed mask overlaps `E_blind`:

```text
Remaining_after = E_blind intersection not(P_visible_before union V_new)
```

For a fixed scene and a strictly cumulative sensor set, adding another peer cannot increase `Remaining`; it either removes some cells or changes nothing. In the live demo, the percentage is not guaranteed to decrease monotonically from frame to frame because:

- actors and occluders move;
- the ego-centred inventory region moves;
- nearest-site selection can replace one site with another;
- selection hysteresis temporarily retains prior sites; and
- entering a network-degraded zone switches from camera/radar pairs to radar-only sites.

More pairs also do not guarantee a reveal. All selected peers may have unsuitable orientation, insufficient range, redundant viewpoints, or blocked sight lines. Selection is nearest-site based, not coverage-optimal.

The camera and radar at one site are evaluated separately because they can have different FoVs and ranges. Since they are co-located, they share the same mount and yaw. Either modality may establish visibility.

## 10. Ego-perspective object classification

For each role-labelled hazard target, the evaluator records observations from active modalities and calculates:

```text
inside_ego_fov = any ego observation has at least one in-FoV probe
visible_to_ego = any ego observation has a clear in-FoV probe
visible_to_peer = any non-ego observation has a clear in-FoV probe

blocked_from_ego = inside_ego_fov and not visible_to_ego
cooperatively_detected = visible_to_peer and not visible_to_ego
network_detected = visible_to_ego or visible_to_peer
```

`seen_by` and `peer_seen_by` retain `(parent_actor_id, modality)` provenance for the successful observations.

The UI reduces these booleans to three states, in priority order:

| State | Condition | Meaning |
|---|---|---|
| `EGO_VISIBLE` | `visible_to_ego` | At least one active ego modality has a clear view. This remains the state even if a peer also sees the target. |
| `COOPERATIVELY_REVEALED` | not ego-visible and `visible_to_peer` | The network knows the target from at least one peer view. The target may be geometrically blocked inside the ego FoV or outside the ego FoV. |
| `UNRESOLVED_OCCLUDED` | neither ego nor peer visibility, stale snapshot, or map disabled | No current active sensor supplies detection authority to the UI. |

From the ego-user-interface perspective, “occluded object” therefore means **peer-visible but ego-not-visible**. This includes both:

- a target behind an opaque object within an ego FoV; and
- a target outside the ego pair's current angular/range coverage but seen by a peer.

`blocked_from_ego` distinguishes the first case internally. The current label does not expose that distinction.

The target's own wearable/mounted pair is always excluded from evidence for that target. Otherwise, activating the target's own pair would make self-detection trivial and invalidate the cooperative-perception demonstration.

## 11. Spatial-map and ego-camera behavior

### 11.1 Spatial map

For role-labelled rogue targets:

| Detection result | Spatial-map behavior |
|---|---|
| No active sensor sees target | Target is withheld to prevent CARLA ground-truth leakage. |
| Ego sensor sees target | Target is drawn using the ordinary actor color. |
| Peer sees target and ego does not | Target is drawn red with `OCCLUDED PEDESTRIAN` or `OCCLUDED VEHICLE`. The pedestrian icon is enlarged. |

Non-rogue actors remain visible as contextual CARLA ground truth. The withholding policy currently applies only to the strict blocker role names.

### 11.2 Visibility snapshot

The top-down renderer stores the per-actor result in a visibility snapshot. The camera path accepts it only while:

- `U` visualization mode is enabled;
- the top-down renderer is available; and
- the snapshot is no more than 0.75 s old.

Otherwise the camera path treats the target as `UNRESOLVED_OCCLUDED` and suppresses its cooperative warning overlay.

### 11.3 Rogue-pedestrian warning authorization

A rogue pedestrian can enter the warning path when its fresh state is either `EGO_VISIBLE` or `COOPERATIVELY_REVEALED`. Additional safety-display gates then apply:

1. The actor role must match the configured pedestrian blocker prefix and positive index.
2. Its planar distance must be inside `--rogue-pedestrian-warning-radius`.
3. It must not have passed behind the ego beyond a clearance derived from both actor bounding radii and a margin.

When those gates pass:

- a projectable pedestrian receives a red bounding box and `PEDESTRIAN` label in the RGB view;
- the nearest qualifying pedestrian controls the top-right warning card;
- distance above `--rogue-pedestrian-brake-radius` produces `SLOW DOWN`; and
- distance at or below the brake radius produces `APPLY BRAKES`.

The warning distance is recorded before RGB bounding-box projection. Therefore the warning card can remain visible when the driver looks away or the close target cannot be projected cleanly, while the red box itself requires a valid image projection.

## 12. Network-degradation interaction

Network degradation and occlusion are separate mechanisms:

- **Occlusion** is geometric and caused by opaque footprints between a sensor and an area/target.
- **Network degradation** is a circular cellular-condition overlay that changes visual sensor scheduling and adds display-only latency/accuracy penalties.

Entering any nonzero zone influence immediately enables radar-priority mode. Leaving requires three clear map refreshes to reduce boundary flicker. In radar-priority mode:

- all virtual cameras are inactive;
- the configured nearest radar sites are active;
- only those radars contribute to object visibility and peer occlusion removal; and
- no virtual sensor is physically started or subscribed.

The switch can increase or decrease geometric coverage because radar has a different FoV/range from camera and because the selected site count can differ.

Network zones may be received from the profile published by `spawn_blocker_v5.py`. Explicit local `--network-degradation-zone X Y RADIUS` options override the shared locations, and `--disable-network-degradation` disables the local penalties and radar-priority policy.

## 13. Worked example

Consider this simplified ground-plane scene:

```text
Ego sensor E = (0, 0), yaw = 0 degrees
Target pedestrian centre T = (20, 0)
Opaque blocker footprint: x in [8, 12], y in [-1, 1]
Peer sensor P = (20, -10), yaw = 90 degrees
Camera FoV = 90 degrees, range = 60 m
```

For the ego centre probe:

```text
d(E,T) = 20 m
bearing(E,T) = 0 degrees
relative angle = 0 degrees
```

The probe is inside the ego camera FoV and range. The segment from `(0,0)` to `(20,0)` crosses the blocker rectangle, so the centre probe is blocked. With a typical pedestrian footprint behind that blocker, all sampled boundary and interior probes also cross it. The ego observation is therefore in-FoV but not visible.

For the peer:

```text
d(P,T) = 10 m
bearing(P,T) = 90 degrees
relative angle = 0 degrees
```

The vertical segment from `(20,-10)` to `(20,0)` does not intersect the blocker at `x=[8,12]`. One clear footprint probe is sufficient, so:

```text
inside_ego_fov = true
visible_to_ego = false
blocked_from_ego = true
visible_to_peer = true
network_detected = true
cooperatively_detected = true
UI state = COOPERATIVELY_REVEALED
```

On the map, the pedestrian appears red with `OCCLUDED PEDESTRIAN`. The peer's unshadowed FoV removes whatever part of the ego blind mask it covers. If the ego vehicle is also within the warning radius and has not passed the pedestrian, the camera warning path becomes authorized.

If the ego later moves far enough that one target-footprint probe gains a clear sight line around the blocker, `visible_to_ego` becomes true. The state changes to `EGO_VISIBLE`, the map uses the ordinary pedestrian color, and the safety warning remains authorized while the proximity/pass gates still apply.

## 14. Configuration and controls

### 14.1 Relevant command-line options

| Option | Purpose | Default |
|---|---|---:|
| `--spatial-map-active-sensor-pairs COUNT` | Complete active sites in normal mode, including ego. | 5 |
| `--spatial-map-degraded-radar-pairs COUNT` | Active radar sites in a degraded zone. | 5 |
| `--spatial-map-sensor-forward-range METERS` | Maximum ego-relative distance for eligible sensor sites. | 40 m |
| `--spatial-map-sensor-forward-half-angle DEGREES` | Half-angle of eligible forward sensor-site sector. | 45 degrees |
| `--cooperative-sensor-pair-horizontal-fov DEGREES` | Set the same FoV on both virtual modalities. | unset |
| `--cooperative-camera-horizontal-fov DEGREES` | Camera-only horizontal FoV. | 90 degrees |
| `--cooperative-camera-range METERS` | Camera visibility range. | 60 m |
| `--cooperative-radar-horizontal-fov DEGREES` | Radar-only horizontal FoV. | 35 degrees |
| `--cooperative-radar-range METERS` | Radar visibility range. | 80 m |
| `--infrastructure-sensor-traffic-light-ids IDS` | Comma-separated virtual infrastructure anchors or `none`. | `14,24,11` |
| `--network-degradation-zone X Y RADIUS` | Repeatable local network-degradation circle. | shared/fallback profile |
| `--disable-network-degradation` | Disable local zone effects. | off |
| `--rogue-pedestrian-warning-radius METERS` | Outer proximity radius for warning overlay. | 45 m |
| `--rogue-pedestrian-brake-radius METERS` | Inner radius for `APPLY BRAKES`. | 12 m |
| `--rogue-pedestrian-role-prefix PREFIX` | Strict blocker role-name stem. | `pedestrian_blocker_v5` |

Example with five selected sites and a common 120-degree FoV for both modalities:

```bash
python3 manual_control_ar_v13.py \
  --topdown-zoom-radius 50 \
  --ego-spawn-x 73.63 --ego-spawn-y 66.36 \
  --vehicle-blueprint vehicle.lincoln.mkz \
  --route-config ./my_ego_route.json \
  --rogue-pedestrian-warning-radius 45 \
  --rogue-pedestrian-brake-radius 12 \
  --spatial-map-active-sensor-pairs 5 \
  --cooperative-sensor-pair-horizontal-fov 120
```

### 14.2 Runtime controls

| Key | Effect |
|---|---|
| `U` | Toggle the spatial map, FoVs, occlusion layer, route, boxes, metrics, and cooperative warnings. |
| `[` or `-` | Reduce the active site budget by one. |
| `]` or `=` | Increase the active site budget by one. |
| Shift plus a count key | Change the budget in steps of five. |

The runtime count adjustment sets the normal pair count and degraded radar count to the same value so the operator controls one visible site budget.

## 15. Current limitations and non-claims

The implementation intentionally has the following limits:

- **2-D only:** no height, pitch, vertical FoV, overpass, slope, or elevation reasoning. A traffic-light pair's 15 m display height has no geometric effect.
- **Ground-truth inputs:** actor transforms, roles, and bounding boxes come directly from CARLA, not from detector output.
- **No physical virtual sensors:** the displayed cooperative camera/radar pairs do not collect images, radar returns, timestamps, or packets.
- **No learned fusion:** camera and radar geometry are combined by boolean OR, without confidence, covariance, temporal alignment, association, or modality weighting.
- **Opaque footprints:** all buildings, vehicles, and pedestrians block the 2-D line regardless of material, height, radar penetrability, or partial transparency.
- **Approximate buildings:** CARLA building bounding boxes are rectangular approximations filtered for road proximity.
- **Finite target sampling:** a very narrow exposed sliver between footprint probes can be missed. “Anywhere” means anywhere covered by this deterministic probe set, not continuous polygon proof.
- **Approximate area mask:** the shadow overlay uses angular quadrilaterals and downsampled raster masks; it can differ slightly from the exact per-object line test.
- **Nearest-site scheduling:** active sites are selected by distance with hysteresis, not by marginal coverage gain or target information value.
- **Hazard-specific gating:** only strict `spawn_blocker_v5.py` rogue roles are withheld until detected. Other map actors remain ground-truth context.
- **Instantaneous state:** there is no tracker or detection latch. The current fresh snapshot controls state, and snapshots older than 0.75 s are rejected.
- **Demonstration warning:** the red box and speed recommendation are presentation logic, not a safety-certified actuation decision.

For a research-grade implementation, the next steps are real timestamped modality observations, 3-D frustum/depth visibility, unknown/free/occupied evidence, cross-sensor association, uncertainty-aware fusion, temporal tracking, ground-truth occlusion masks, and precision/recall plus warning-timeliness evaluation.

## 16. Troubleshooting interpretation

| Symptom | Likely interpretation/check |
|---|---|
| Rogue target never appears | Enable `U`; confirm a fresh map snapshot, positive active count, strict role name, eligible peer site, sensor FoV/range, and at least one clear footprint probe. |
| Increasing from one to five pairs changes nothing | The added nearest sites may be outside useful orientation/range, mutually occluded, redundant, or the target's own excluded site. Widening sensor FoV does not widen the host inventory sector. |
| Target is red on map but has no red camera box | Peer visibility authorized it, but the RGB bounding box may be outside the camera image or fail projection. Check warning distance and pass status too. |
| Warning card appears without a box | This is allowed: proximity is evaluated before image projection so the warning survives a driver look-away. |
| Purple region and target state disagree near an edge | The purple layer is downsampled shadow geometry; the target state uses exact segments to footprint probes. |
| Coverage changes at a network-zone boundary | Radar-only policy entered immediately or exited after its three-refresh debounce. This is not an occluder appearing/disappearing. |
| A visible ordinary actor was never sensor-detected | Non-rogue actors are currently displayed as CARLA ground-truth context; only role-labelled hazards use the detection gate. |

## 17. Implementation map and validation

| Responsibility | File and symbol |
|---|---|
| Actor footprint corners | `manual_control_ar_v13.py`: `get_actor_footprint_points()` |
| Virtual sites and selection | `manual_control_ar_v13.py`: `_virtual_sensor_pair()`, `_prepare_sensor_markers()`, `_apply_visual_sensor_policy()` |
| Live occluder/target snapshot | `manual_control_ar_v13.py`: `_collect_live_scene()` |
| Area occlusion overlay | `manual_control_ar_v13.py`: `_draw_cooperative_visibility_layer()` |
| Map object withholding/labels | `manual_control_ar_v13.py`: `_draw_live_actors()` |
| Snapshot state gate | `manual_control_ar_v13.py`: `cooperative_visibility_state()` |
| Camera boxes and warning proximity | `manual_control_ar_v13.py`: `_draw_actor_bboxes()` |
| FoV, shadow, line-of-sight, and target fusion | `cooperative_occlusion_v1.py` |
| Offline regression coverage | `test_manual_control_ar_v13_occlusion.py` |

The offline tests exercise FoV boundaries, shadow geometry, target-footprint partial exposure, occluder intersections, self-sensor exclusion, ego/peer fusion, stale-state handling, warning authorization, rendering gates, CLI validation, and degraded radar behavior. Run them from `neu_collab` with:

```bash
/home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/python \
  -m unittest -v test_manual_control_ar_v13_occlusion.py
```

Passing offline tests verifies deterministic helper and mocked integration behavior. A live Town10HD run remains necessary to validate visual placement, actor timing, performance, and the complete `spawn_blocker_v5.py` interaction.
