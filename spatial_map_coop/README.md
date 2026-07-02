# Cooperative Spatial Map — moving-ego build (incremental)

Building the cooperative-perception spatial map **bit by bit**, retiring risk in order.
Fancy algorithms (JPDA / EKF / covariance-intersection fusion, frustum occlusion deduction,
alert feedback) are deliberately deferred until the plain multi-car map works.

Model: **200k-pps RGB+radar fusion** (accuracy sweet spot; zero extra transport cost — see
`../PPS_STUDY_SUMMARY.md`).

## Stages
1. **Moving-ego dynamic ROI** *(current)* — one ego drives; the top-down map ROI follows the car
   instead of cropping to a fixed traffic-light pole. Objects placed as today (no fusion changes).
   - Open question — ROI size: start with a **fixed box = the model's detection range (~40 m,
     the ≤40 m detection gate)**, optionally forward-biased along heading (`--focus-follow-forward-bias`).
     Speed-adaptive ROI is a later refinement.
2. **Two egos, no fusion** — spawn a second ego behind the first, same area, both drive. Render
   each car's detections in its **own color** (Car A blue, Car B another), no association/fusion yet.
3. **Geometry fusion / frustum / occlusion** — only after 1–2 are solid.

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
  --focus-follow-forward-bias 0.35
# viewer: http://127.0.0.1:35011/api/spatial_map/viewer
```

**Terminal 3 — driving ego (sensors on the car, drives a fixed loop, streams objects):**
```bash
cd /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
source /home/shr_aisvcs/workarea/carla_0_10_env/carla_0_10_venv/bin/activate
export PYTHONPATH="$PWD/pole_lraspp_multimodal_fusion:$PWD:${PYTHONPATH:-}"
python3 carla_split_inference_udp_fusion_object_pole_client_spatial_stream_oai.py \
  --role loopback --headless \
  --sensor-platform ego_vehicle \
  --ego-fixed-path-spawn-indices 80,85,91,94,99,80 --ego-fixed-path-loop \
  --fusion-checkpoint experiments/autonomous_arch_runs_20260625/det_pps200000_v2/checkpoints/det_pps200000_v2/best.pt \
  --npc-vehicles 28 --npc-pedestrians 35 --spawn-radius 80 \
  --spatial-map-stream-id fusion_ego --spatial-map-port 39201 \
  --result-timeout 0.15
```
The map ROI should now travel with the car and objects should appear within ~40 m of it.
Do **not** pass `--focus-traffic-light-ids 14` to the server for this stage (that crops to the pole).
