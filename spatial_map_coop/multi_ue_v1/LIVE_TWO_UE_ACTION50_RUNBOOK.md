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
