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

## 2. Start a moving two-vehicle CARLA scenario

Start CARLA and the existing Stage-2 moving two-ego scenario described in
`spatial_map_coop/README.md`. Those clients own motion; this harness owns only
its attached RGB/radar sensors. A different scenario is also valid provided
two distinct moving vehicle actors observe some common vehicles or pedestrians.

List the live vehicle identities without loading CUDA or a model:

```bash
python3 -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --list-vehicles --carla-host 127.0.0.1 --carla-port 2000
```

Select the two actor IDs belonging to the egos. Do not choose NPC IDs. Their
`role_name`, type and location are included to make the selection auditable.

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

Choose a new, absent output leaf and substitute the two actor IDs:

```bash
python3 -u -m spatial_map_coop.multi_ue_v1.live_two_ue_action50_v1 \
  --execute SPLITFUSION_LIVE_TWO_UE_ACTION50_DEMO \
  --ue-a-actor-id UE_A_ID \
  --ue-b-actor-id UE_B_ID \
  --max-pairs 20 \
  --duration-s 180 \
  --output experiments/spatial_map_multi_ue_v1/live_two_ue_action50_run1
```

The harness selects common CARLA frame IDs, uses a two-tick stride, and runs
the two sources sequentially through one resident RTX model stack. Therefore
its `localhost_model_path_ms` is diagnostic compute time, not an end-to-end or
parallel-UE latency claim.

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

## 5. Return evidence for review

The output contains only:

- `observations.jsonl` — per-UE model/payload/compute summaries;
- `snapshots.jsonl` — associated objects and provenance-preserving tracks;
- `faults.jsonl` — isolated shadow-fault results;
- `SUMMARY.json`, `manifest.json`, and the success terminal.

No raw RGB, radar tensors or semantic masks are stored. Preserve the directory
and send it back without editing. Git may ignore `experiments/`, so transfer it
explicitly rather than assuming `git push` includes it.

## Scientific interpretation

A pass demonstrates that two live UEs can deliver real learned detections into
one map, that compatible observations are associated one-to-one, and that the
selected state retains complete source/hash provenance while invalid, stale and
conflicting inputs are handled explicitly. It does not yet prove covariance-
weighted fusion, persistent M-of-N track confirmation, occlusion reasoning,
agent-selected actions, UI behavior or network performance.
