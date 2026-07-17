# Cooperative Spatial Map — moving-ego build (incremental)

**Status reconciled 2026-07-16.** The moving-ego view, two-source live display, offline replay,
and a synthetic field-of-view occlusion baseline are working. The map is still a per-source
visualization: it does not yet perform real multi-view association/fusion or issue validated warnings.

Building the cooperative-perception spatial map **bit by bit**, retiring risk in order.
Fancy algorithms (JPDA / EKF / covariance-intersection fusion, frustum occlusion deduction,
alert feedback) are deliberately deferred until the plain multi-car map works.

Model: **200k-pps RGB+radar fusion** (accuracy sweet spot; zero extra transport cost — see
`../PPS_STUDY_SUMMARY.md`).

## Stages

1. **Moving-ego dynamic ROI — complete.** One ego drives; the top-down map ROI follows the car
   instead of cropping to a fixed traffic-light pole. Objects placed as today (no fusion changes).
   - Open question — ROI size: start with a **fixed box = the model's detection range (~40 m,
     the ≤40 m detection gate)**, optionally forward-biased along heading (`--focus-follow-forward-bias`).
     Speed-adaptive ROI is a later refinement.
2. **Two egos, no fusion — complete.** Spawn a second ego behind the first, same area, both drive. Render
   each car's detections in its **own color** (Car A blue, Car B another), no association/fusion yet.
3. **Replay + synthetic occlusion baseline — complete.** Recorded/synthetic scenes can be replayed
   offline; FoV-membership reasoning passes the known synthetic truck/pedestrian scene.
4. **Real cooperative reasoning — current research gap.** Add cross-source association and fusion,
   ray/visibility-grid occlusion disambiguation, real CARLA ground truth, precision/recall evaluation,
   then the vehicle warning/feedback loop.

## What changed vs the baseline server
`spatial_map_server_moving_ego.py` is a copy of `../../real_time_spatial_map_server_fusion_object_v2.py`
(per the "never edit top-level scripts" convention) with a **follow-ego ROI**:
- retains the incoming packet's `anchor.transform` (the ego's live pose),
- `--focus-follow-stream-id <id>`: centers the ROI on that stream's current ego pose, following it
  as it drives (takes precedence over `--focus-traffic-light-ids`),
- `--focus-follow-forward-bias <f>`: push the ROI ahead of the car by `f × radius`.

## Stage-1 run recipe
Three terminals (CARLA started manually first). Ports match the existing runbook.

**Terminal 1 — CARLA** (user starts; Town10HD).

**Terminal 2 — map server (follow the ego, 40 m ROI):**
```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
python3 abiodun/spatial_map_coop/spatial_map_server_moving_ego.py \
  --focus-follow-stream-id fusion_ego \
  --focus-radius-m 40 \
  --focus-follow-forward-bias 0.35 \
  --stream-stale-s 10   # tolerate the slower non-headless client cadence (default 2.5s blanks the map)
# viewer (live HTML5 canvas, ego-centered, ~10Hz): http://127.0.0.1:35011/api/spatial_map/viewer
# fallback (old matplotlib PNG):                     http://127.0.0.1:35011/api/spatial_map/viewer_png
```

**Terminal 3 — driving ego (sensors on the car, drives the *trained* loop, streams objects):**
```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
export PYTHONPATH="$PWD/pole_lraspp_multimodal_fusion:$PWD:${PYTHONPATH:-}"
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role loopback \
  --sensor-platform ego_vehicle --no-ego-freeze \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m 0 --ego-spawn-right-offset-m 0 --ego-spawn-z-offset-m 0.15 \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop \
  --ego-disable-lane-change --ego-ignore-lights-pct 50 \
  --no-draw-projected-obb-box \
  --fusion-checkpoint experiments/autonomous_arch_runs_20260625/det_pps200000_v2/checkpoints/det_pps200000_v2/best.pt \
  --seed 31 --npc-vehicles 28 --npc-pedestrians 35 --spawn-radius 80 \
  --spatial-map-stream-id fusion_ego --spatial-map-port 39201 \
  --result-timeout 1.5 --headless
```
Note: `--result-timeout 1.5` (not 0.15). The client publishes a map packet only on frames whose
loopback result returns; at 0.15 s ~80% time out, so the stream starves and the map goes stale. 1.5 s
is the live-demo value. (0.15 was only for the fast payload/latency cost sweep.)
Key fixes vs the first attempt:
- **`--no-ego-freeze`** — otherwise the ego stays parked (`--ego-freeze` defaults True).
- **`--ego-spawn-index 80`** — spawns on-lane at the first route waypoint (the trained area). Default
  `-1` picks a random spawn point in an area the model never saw.
- offsets 0 so it spawns *on the lane*, not pushed to the curb (curb offsets are for the parked-ego demo).

The map ROI should travel with the car; vehicle/pedestrian boxes appear within ~40 m of it.
Do **not** pass `--focus-traffic-light-ids 14` to the server for this stage (that crops to the pole).

## Stage 2 — second (trailing) ego, colored by source
Keep Terminal 1–3 as above (car A = `fusion_ego`). Add a **Terminal 4** for car B, with a **distinct
stream-id and distinct internal ports** (two clients on one host must not share ports), trailing A on the
same loop:
```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
export PYTHONPATH="$PWD/pole_lraspp_multimodal_fusion:$PWD:${PYTHONPATH:-}"
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role loopback --headless \
  --sensor-platform ego_vehicle --no-ego-freeze \
  --ego-spawn-index 80 \
  --ego-spawn-forward-offset-m -15 --ego-spawn-right-offset-m 0 --ego-spawn-z-offset-m 0.15 \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop \
  --ego-disable-lane-change --ego-ignore-lights-pct 50 --no-draw-projected-obb-box \
  --fusion-checkpoint experiments/autonomous_arch_runs_20260625/det_pps200000_v2/checkpoints/det_pps200000_v2/best.pt \
  --seed 31 --npc-vehicles 28 --npc-pedestrians 35 --spawn-radius 80 \
  --spatial-map-stream-id fusion_ego_b --spatial-map-port 39201 \
  --camera-source-port 51101 --remote-port 51102 --remote-source-port 51103 --camera-result-port 51104 \
  --result-timeout 1.5
```
- B spawns at the **same route start (index 80)** but `--ego-spawn-forward-offset-m -15` places it
  **~15 m behind car A** on the same lane. Both drive the identical loop, so they view the *same* objects
  at *different closeness* (the cooperative-perception premise). Tune -10…-20 for the gap.
- Distinct ports `511xx` vs A's default `510xx`; distinct `--spatial-map-stream-id fusion_ego_b`.
- The **canvas auto-switches to color-by-source** once a 2nd stream appears: car A and car B get
  different colors with a legend (top-right). No fusion/association — each car's raw detections shown
  as-is (the intended Stage-2 "before" picture). ROI keeps following `fusion_ego` (car A).
- If the two clients fight over world ticking, make A `--sync-world` and B `--async-world`.

## Troubleshooting Stage 1
Watch the client's stdout first: it should print `Moving ego vehicle: ... freeze=False` and
`Moving ego autopilot enabled: ... fixed_path_points=N` (N>0). If it says `Parked ego vehicle`, the
`--no-ego-freeze` flag didn't take.

Then probe the server while the client runs:
```bash
# Are packets arriving from the ego? (expect stream_id=fusion_ego, object_count>0 once it detects)
curl -s http://127.0.0.1:35011/api/fusion_streams/latest | python3 -m json.tool | head -40
# Is the follow-ROI locking onto the ego pose? (metadata.focus_view.mode == "follow_stream", ego_pose set)
curl -s http://127.0.0.1:35011/api/spatial_map/latest | python3 -c 'import json,sys; d=json.load(sys.stdin); print("focus:",d["metadata"]["focus_view"]); print("n_objects:",len(d["spatial_map_objects"]))'
```
- `fusion_streams` empty → client isn't publishing (check `--spatial-map-port 39201` on both, and that the
  client says spatial map enabled).
- stream present but `object_count=0` → model detecting nothing (wrong area / not driving yet / too few NPCs).
- `focus_view.mode != "follow_stream"` or `warning` about the stream → `--focus-follow-stream-id` on the
  server doesn't match `--spatial-map-stream-id` on the client (both must be `fusion_ego`).
