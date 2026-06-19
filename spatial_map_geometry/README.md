# SceneSense Spatial Map Geometry Scaffold

Purpose: build the shared-map reasoning layer gradually, without mixing early
experiments into the existing live spatial-map server.

The core idea is:

1. Each car, pole, or camera publishes a local sensor map.
2. A local sensor map contains the sensor pose, visible ground footprint, and
   objects detected in world coordinates.
3. The spatial-map server overlays local maps from multiple viewpoints.
4. Overlap between visible footprints is used to reason about shared objects,
   missed detections, occlusion hypotheses, freshness, and provenance.

This folder is intentionally a starter scaffold. It is not yet the final map
server. The files here should let us prototype the geometry and logic offline
before wiring it into live CARLA streams.

## Coordinate Convention

For the first prototype, all top-down geometry is in CARLA/world meters:

- `x`: world x coordinate
- `y`: world y coordinate
- `yaw_deg`: heading angle in degrees, where `0` points along positive x
- FoV polygons are 2D ground-plane approximations of camera/radar visibility

This is sufficient for first-pass reasoning. Later we can add height-aware
visibility, camera projection, true 3D frustums, and ray-based occlusion.

## Files

- `schemas.py`: dataclasses for local maps, objects, associations, and
  occlusion hypotheses.
- `geometry.py`: FoV polygon construction, polygon overlap, point-in-polygon,
  and object footprint helpers.
- `association.py`: simple cross-map object association using class and XY
  distance.
- `occlusion_reasoner.py`: first-pass overlap/disagreement reasoning.
- `demo_two_view_overlap.py`: offline demo with two local maps, overlapping
  FoVs, shared objects, and disagreement hypotheses.
- `live_visibility_server.py`: standalone Flask prototype that accepts local
  sensor-map payloads over HTTP and renders a live top-down visibility map.

## Development Stages

### Stage 0: Offline Geometry Scaffold

Status: started in this folder.

Completion criteria:

- Create local map/object schemas.
- Compute camera/radar FoV ground footprints.
- Compute overlap between two sensor footprints.
- Associate objects from two maps by class and world XY distance.
- Generate first-pass "seen by A, missing from B" hypotheses.

### Stage 1: Two-View CARLA Sanity Demo

Goal: place two parked ego vehicles or pole cameras with overlapping FoVs and
one object in the shared region.

Completion criteria:

- Export a local map JSON for each viewpoint.
- Draw both FoVs, objects, overlap region, and association lines.
- Verify that shared objects are associated and missing objects are flagged.
- Confirm timestamps/freshness prevent stale maps from causing false warnings.

### Stage 2: Connect To Existing Fusion Runtime

Goal: extend the current fusion clients to publish local-map metadata in
addition to object predictions.

Required fields:

- `stream_id`
- `sensor_pose`
- `fov_polygon`
- `camera_fov_deg`
- `range_m`
- `objects`
- `timestamp_s`
- `frame_id`
- `model_name`
- `provenance`

Completion criteria:

- Existing spatial-map server can still receive old object-only payloads.
- New payloads include FoV footprint and source pose.
- Offline scripts can replay logged local maps without CARLA.

### Stage 3: Overlap And Occlusion Reasoning

Goal: turn overlap/disagreement into explainable hypotheses.

Reasoning examples:

- Object appears in A and B: common object, higher confidence.
- Object appears in A, lies inside B footprint, but is missing from B:
  possible occlusion, B detection miss, stale B map, or A false positive.
- Object appears in A but outside B footprint: no warning to B.

Completion criteria:

- Hypotheses include source stream, target stream, object id/class, overlap
  area, object location, freshness, and reason.
- Output is conservative: say "possible occlusion" rather than "guaranteed."

### Stage 4: Live Server Integration

Goal: add this reasoning to the live spatial-map service.

Options:

- Keep `real_time_spatial_map_server_fusion_object_v2.py` unchanged.
- Build a small standalone server first, then merge ideas once the API
  stabilizes.

Completion criteria:

- `/api/spatial_map/latest` includes visibility footprints.
- Viewer draws sensor FoVs and overlap regions.
- Viewer marks common objects and possible occluded/missed objects.

Starter command:

```bash
cd abiodun
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 spatial_map_geometry/live_visibility_server.py \
  --api-port 35021 \
  --render-interval-ms 500 \
  --load-demo-on-start
```

Open:

```text
http://127.0.0.1:35021/api/spatial_map/viewer
http://127.0.0.1:35021/api/spatial_map/viewer?demo=1
http://127.0.0.1:35021/api/spatial_map/latest
```

The `?demo=1` form asks the running server to load the two synthetic demo
sensor maps from the browser. The viewer header also shows status, active
stream count, pinned demo streams, and any render error.

If CARLA is unavailable or you want a clean geometry-only view:

```bash
MPLCONFIGDIR=/tmp/matplotlib-cache \
python3 spatial_map_geometry/live_visibility_server.py \
  --api-port 35021 \
  --render-interval-ms 500 \
  --no-carla-static-map \
  --load-demo-on-start
```

Demo streams loaded with `--load-demo-on-start` are pinned by default, so they
do not disappear after `--stream-stale-s`. For live streams posted to
`/api/local_maps/update`, stale expiry still applies.

The viewer intentionally limits itself to one in-flight PNG request at a time.
Matplotlib rendering is too heavy for true 100 ms browser polling; later, the
real spatial-map UI should use a canvas/WebSocket renderer for that cadence.

Troubleshooting:

- If the viewer says `waiting_for_local_maps`, load demo maps with
  `/api/local_maps/demo`, start with `--load-demo-on-start`, or post a local-map
  payload to `/api/local_maps/update`.
- If the browser polls `/api/spatial_map/live.png` with HTTP 200 but the image
  appears blank, check whether maps are loaded:

```bash
curl -s http://127.0.0.1:35021/healthz | python3 -m json.tool
curl -s -X POST http://127.0.0.1:35021/api/local_maps/demo | python3 -m json.tool
curl -s http://127.0.0.1:35021/api/spatial_map/latest | python3 -m json.tool
curl -s -o /tmp/scenesense_visibility_map.png -w '%{http_code} %{size_download}\n' \
  'http://127.0.0.1:35021/api/spatial_map/live.png?force=1'
file /tmp/scenesense_visibility_map.png
```

Expected after the demo load: `streams` should include `car_A` and `car_B`,
`active_stream_count` should be `2`, and the PNG size should be larger than a
few kilobytes.
- If the viewer says `Static-map fallback: Connection refused`, the server
  process cannot reach CARLA at `--carla-host/--carla-port`. Confirm from the
  same machine:

```bash
python3 - <<'PY'
import carla
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(2.0)
world = client.get_world()
print(world.get_map().name)
PY
```

If this fails, either CARLA is running on another host/port or the Python
environment cannot import/connect to CARLA. Pass the correct `--carla-host` and
`--carla-port`, or run the visibility server on the same machine as CARLA.

Payload endpoint:

```text
POST /api/local_maps/update
```

Accepted payload shape:

```json
{
  "stream_id": "car_A",
  "pose": {"x": 0.0, "y": 0.0, "z": 1.5, "yaw_deg": 25.0},
  "fov_deg": 90.0,
  "range_m": 60.0,
  "objects": [
    {
      "id": "car_A_obj_001",
      "class_name": "vehicle",
      "location": {"x": 18.0, "y": 3.0, "z": 0.0},
      "dimensions": {"length": 4.5, "width": 1.9, "height": 1.7},
      "yaw_deg": 10.0,
      "score": 0.88
    }
  ]
}
```

### Stage 5: Evaluation

Goal: quantify whether overlap-aware sharing improves cooperative perception.

Metrics:

- shared-object association precision/recall
- occlusion warning precision/recall
- stale warning rate
- false hazard rate
- object localization error
- time-to-warning
- payload cost per local map

### Stage 6: RL/Controller Integration

Goal: use map state as part of the agent state and action context.

Candidate state features:

- number of high-risk overlap disagreements
- object freshness
- occlusion probability
- source confidence/provenance
- local scene density
- network load and payload budget

Candidate actions:

- request update from a specific UE
- ask UE to send OD instead of SEG
- ask UE to increase/decrease map detail
- change send rate
- suppress redundant map updates

## Notes

- FoV overlap does not mean both sensors must detect the same object. The object
  may be occluded from one viewpoint, too small, stale, outside vertical FoV, or
  below threshold.
- CARLA ground truth can be used for labels and evaluation. It should not be
  treated as available to the deployable runtime.
- Start with explainable geometry before adding learned occlusion models.
