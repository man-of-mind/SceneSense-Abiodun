# L10319 live two-UE/action-50 runbook

This is a headless functional demonstration of two real CARLA sensor streams,
the exact locked SplitFusion action 50, and the conservative multi-UE spatial
map. It is not a UI task and does not measure OAI or two-device latency.

## 1. Obtain the branch

From the repository on this machine:

```bash
git fetch origin
git switch spatial-map-coop-v1
git pull --ff-only origin spatial-map-coop-v1
```

Do not merge it into the campaign branch while the 288-cell run is active.
Do not export `PYTHONPATH`; activate the registered CARLA environment and run
commands from the `abiodun` repository root.

## 2. Start a fresh CARLA server

Start CARLA on port 2000 with Town10HD_Opt, but do not start the two historical
Stage-2 inference clients. The primary run mode owns the two egos, one CARLA
clock, background population, and only the four sensors required by this
demonstration. This avoids loading three perception pipelines or relying on
manually copied actor IDs from an earlier CARLA lifecycle.

The CARLA world must initially contain no vehicles or pedestrians. Static map
actors are expected and allowed.

The older passive-attachment mode remains available for a separately managed
scenario. Its live vehicle identities can be listed without loading CUDA:

List the live vehicle identities without loading CUDA or a model:

```bash
python3 -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --list-vehicles --carla-host 127.0.0.1 --carla-port 2000
```

Never use an ID printed before a CARLA restart or after its owning client
exits. CARLA actor IDs are lifecycle-local.

## 3. Preflight the locked profile

```bash
python3 -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 --preflight
```

This must report action 50 as
`split_ae64_uint4_q5000`, `AE64`, `UINT4`, q=0.50, keep count 10,752,
`CURRENT_CELL_MAJOR`, zstd level 1, and a visible CUDA device. It also verifies
the runtime catalog and all registered artifact hashes. If a bound gitignored
checkpoint/evidence artifact is absent on L10319, stop and copy that immutable
artifact from the qualified machine; never weaken the hash check.

## 4. Run once

Choose a new, absent output leaf. The primary self-contained command is:

```bash
python3 -u -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --execute SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO \
  --spawn-two-egos \
  --ego-spawn-index 80 --ego-gap-m 15 \
  --npc-vehicles 28 --npc-pedestrians 35 \
  --max-pairs 20 \
  --duration-s 180 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_run1
```

This mode configures the registered 20 Hz synchronous CARLA/sensor clock,
assigns distinct UE role names, drives both egos on the registered Stage-2
path, and destroys only the actors it created before restoring the original
CARLA settings. The harness selects common CARLA frame IDs and prepares the
model every two ticks at 10 Hz, preserving the required four radar callbacks
across two 100 ms logical sweeps. It runs the two sources sequentially through
one resident RTX model stack. Therefore its
`localhost_model_path_ms` is diagnostic compute time, not an end-to-end or
parallel-UE latency claim.

For passive attachment instead, omit `--spawn-two-egos` and supply two IDs
that are present in the immediately preceding `--list-vehicles` result.

Success requires:

- 20 paired frames / 40 real model observation batches;
- all five shadow faults exercised;
- strict rejection of non-finite and conflicting identity payloads;
- idempotent duplicate handling and explicit stale filtering;
- at least one nominal two-source association;
- `SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO_COMPLETE`.

The +8 m finite observation is expected not to match its true counterpart. It
may remain visible as a separate single-source track; this is a documented v1
limitation, not evidence that confidence is positional uncertainty.

If no two-source association occurs, the run fails honestly: the chosen egos
did not jointly observe a compatible object. Reposition the scenario or choose
a segment with overlapping views, then use a new create-only output leaf.

If the run fails for another reason, `FAILED.json` includes compact per-UE
camera/radar callback, logical-sweep, preparation-rejection and progress
diagnostics. Preserve it and use a new create-only output leaf after review.

## 5. Return evidence for review

The output contains only:

- `observations.jsonl` — per-UE model/payload/compute summaries;
- `snapshots.jsonl` — associated objects and provenance-preserving tracks;
- `faults.jsonl` — isolated shadow-fault results;
- `SUMMARY.json`, `manifest.json`, and the success terminal.

No raw RGB, radar tensors or semantic masks are stored. Preserve the directory
and send it back without editing. Git may ignore `experiments/`, so transfer it
explicitly rather than assuming `git push` includes it.

## 6. Render the global diagnostic

The original completed evidence can be visualized offline as a global diagnostic;
do not rerun CARLA merely to regenerate this view. From the isolated worktree,
use an absent output leaf:

```bash
python3 -m spatial_map_coop.multi_ue_v1.render_action50_evidence_v1 \
  --run-dir experiments/spatial_map_multi_ue_v1/live_two_ue_action50_self_contained_run2 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_visualization_v1 \
  --fps 15 --interpolation-steps 5
```

This creates 20 measured PNG keyframes, a smoothly interpolated MP4, and a
manifest bound to the successful source evidence and Town10HD static geometry.
Stable track IDs move continuously; new and expired tracks fade instead of
blinking. Cyan/orange indicates the selected UE observation and green identifies
a track supported by both UEs. Interpolation is display-only and never changes
the measured track inventory. The result is a global world-frame technology
diagram, not the final advisor demo, a camera view, CARLA ground truth, or a UI
qualification.

## 7. Collect and render the advisor before/after demonstration

The global diagnostic does not retain the exact per-UE reports or ego poses
needed to explain association visually. After pulling the presentation update
on L10319, run the same short live demonstration once with a new output leaf:

```bash
python3 -u -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --execute SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO \
  --spawn-two-egos \
  --ego-spawn-index 80 --ego-gap-m 15 \
  --npc-vehicles 28 --npc-pedestrians 35 \
  --max-pairs 20 --duration-s 180 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_presentation_run3
```

This still retains no RGB, radar tensor, semantic mask or CARLA actor ground
truth. It adds only two measured ego poses, both compact model-object report
sets and their association identities to each of the 20 snapshot rows.

Render the result offline:

```bash
python3 -m spatial_map_coop.multi_ue_v1.render_action50_before_after_v2 \
  --run-dir experiments/spatial_map_multi_ue_v1/live_two_ue_action50_presentation_run3 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_before_after_v2 \
  --fps 15 --prediction-steps 8 --focus-radius-m 40 --forward-bias 0.35
```

The left panel is the counterfactual presentation view before cross-UE
association: cyan UE-A and orange UE-B reports can overlap as duplicates. The
right panel is the actual conservative map: cyan means A-only, orange B-only,
and green A+B support represented as one track. Valid unmatched objects remain.

Both panels follow UE-A within a 40 m radius and retain both ego markers.
Vehicles use canonical display dimensions and the nearest Town10HD road tangent
for display yaw, while pedestrians are not lane-aligned. No world XY value is
snapped to a lane and lane geometry does not affect scientific association.
Between measured frames the right panel uses a bounded, causal constant-velocity
display predictor; the following measurement corrects it. That predictor is
zero-learning presentation logic and never modifies the evidence or map state.

## Scientific interpretation

A pass demonstrates that two live UEs can deliver real learned detections into
one map, that compatible observations are associated one-to-one, and that the
selected state retains complete source/hash provenance while invalid, stale and
conflicting inputs are handled explicitly. It does not yet prove covariance-
weighted fusion, persistent M-of-N track confirmation, occlusion reasoning,
agent-selected actions, UI behavior or network performance.

## 8. Record the live raw-versus-associated advisor demonstration

This is the preferred presentation path. It uses one experiment and the existing
Stage-2 browser map; the keyboard changes only what is drawn. It does not switch
the association algorithm on or off inside the scientific service and therefore
cannot alter the observations, tracks, counters, or retained evidence.

Start a fresh CARLA server on port 2000. Do not launch the historical Stage-2 UE
clients. In terminal 1, start the map service:

```bash
python3 -u spatial_map_coop/spatial_map_server_moving_ego.py \
  --focus-follow-stream-id two-ue/ue-a \
  --focus-radius-m 40 --focus-padding-m 0 \
  --focus-follow-forward-bias 0.35 \
  --render-hz 0.1 \
  --stream-stale-s 10 \
  --multi-ue-v1 \
  --multi-ue-session-id two-ue-live-action50-v1 \
  --multi-ue-source two-ue/ue-a=ue-a \
  --multi-ue-source two-ue/ue-b=ue-b \
  --output-dir /tmp/spatial_map_action50_live_view
```

Open `http://127.0.0.1:35011/api/spatial_map/viewer`. The default view is the
associated map. Press `R` or `1` for raw reports, `F` or `2` for associated
tracks, or Space to toggle. Buttons at the lower-right provide the same control.
Raw mode shows the two UE report sets independently. Associated mode shows A-only
and B-only tracks in their source colors and one green representation for an
A+B association. A lane tangent may orient nearby vehicle rectangles for display,
but it never changes their measured world-XY positions or scientific association.

In terminal 2, run a longer single demonstration using a new output leaf:

```bash
python3 -u -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --execute SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO \
  --spawn-two-egos \
  --ego-spawn-index 80 --ego-gap-m 15 \
  --npc-vehicles 16 --npc-pedestrians 8 \
  --max-pairs 80 --duration-s 300 \
  --live-map-url http://127.0.0.1:35011 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_live_view_run6
```

The harness preflights the live-map session and source identities before creating
its output. Every accepted UE update is counted in `SUMMARY.json`. With 80 paired
frames the recorded portion should last roughly 90 seconds on the qualified
L10319 setup, although the duration is bounded by measured completion rather than
claimed as a real-time rate. The experiment terminates by itself; stop the map
server separately with Ctrl+C after recording.

The owned synchronous CARLA clock advances in its own 20 Hz thread rather than
waiting for the two sequential model paths. The browser canvas redraws at the
display refresh rate and requests snapshots at 20 Hz; new learned detections still
arrive only when both model paths finish. Display interpolation makes those
measured steps readable without inventing additional detections. The translucent
cyan/orange wedges are the two measured 120-degree sensor FoVs. The associated
presentation renders only tracks supported by the current association snapshot;
historical tracks remain unchanged in the scientific service state and evidence.
