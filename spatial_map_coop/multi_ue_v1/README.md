# Multi-UE cooperative-map ingress v1

Status: **initial filtering/association/server integration; live visual qualification pending**.

This package is the first production-facing Task-2 layer. It validates compact
world-frame object updates from multiple UE streams, retains a bounded history,
selects causal source frames, filters inadmissible observations, associates
compatible observations with one-to-one Hungarian assignment, and maintains a
conservative spatial-map track inventory.

It accepts two existing formats through explicit adapters:

- the current `splitfusion_edge_result.v2` containing
  `splitfusion_object_map_update.v1`; and
- the historical `fusion_object_spatial_map.v1` two-ego packet.

The caller must provide an explicit stable `ue_id` and `session_id`. Stream ID
alone is not treated as vehicle identity. The two adapters also use different
clock-domain labels: their times must not be mixed silently. Capture and
receipt clocks are named separately because the legacy CARLA simulation clock
cannot be subtracted from the map server's Unix receipt clock. Cross-UE AoI is
only meaningful after a shared clock or a measured clock-offset contract is
bound.

## Filtering and association contract

The v1 path applies these operations in order:

1. schema, source/session/stream/frame identity, finite-value, positive-size,
   and duplicate/conflict validation;
2. configurable AoI and cross-source capture-alignment filtering;
3. canonical class compatibility, configurable world-XY distance, dimension,
   and pair-time gates;
4. deterministic sequential class-gated Hungarian assignment, with at most
   one observation from each source in an associated group; and
5. persistent track matching with a bounded stale lifetime.

Position is **not averaged**. The freshest associated observation supplies the
track state and detector confidence is only the tie-breaker for equal capture
times. Every output retains the contributing UE/session/stream identities and
the SHA-256 of every original object record. Confidence is never presented as
positional uncertainty.

The default live-server score floor remains zero: the current SplitFusion
source is already the deployed p025 output, while the legacy adapter preserves
that source's own upstream filtering. Nonzero `--min-object-score` is labelled
an operator demo setting, not calibrated uncertainty. Likewise, the initial
age, alignment, distance, and size values are engineering settings awaiting
independent validation; they are not new scientific gates.

`MultiUESpatialMapService` exposes the path directly and
`create_flask_app()` supplies versioned HTTP ingress and map endpoints. The
existing `spatial_map_server_moving_ego.py` can use the same engine for its
legacy UDP clients via `--multi-ue-v1` plus explicit `STREAM_ID=UE_ID`
bindings. Agent policy is explicitly reported as pending.

## Deterministic two-UE check

The CARLA-free demonstration sends two close observations from different UEs
and requires one track with two-source provenance and the fresher source's
unaltered location:

```bash
python3 -m spatial_map_coop.multi_ue_v1.demo_two_ue_v1
```

The existing two-ego CARLA scenario can exercise the live visual path without
creating a new route. Start its server with the additional arguments:

```bash
python3 spatial_map_coop/spatial_map_server_moving_ego.py \
  --focus-follow-stream-id fusion_ego \
  --focus-radius-m 40 --focus-follow-forward-bias 0.35 \
  --stream-stale-s 10 \
  --multi-ue-v1 \
  --multi-ue-session-id two-ego-demo-1 \
  --multi-ue-source fusion_ego=ue-a \
  --multi-ue-source fusion_ego_b=ue-b
```

Then use the existing Stage-2 car-A and car-B commands in
`spatial_map_coop/README.md`. A successful live check must show both registered
UE streams and at least one two-source track in `/api/spatial_map/latest`.

SciPy is required for `scipy.optimize.linear_sum_assignment`; both the system
Python and the registered CARLA virtual environment currently provide it.

## Deliberate boundary

This version does not estimate positional covariance, average positions,
perform JPDA, classify occlusion from learned tracks, or send downlink warnings.
The existing greedy confidence-weighted fusion remains historical evidence,
not an implicit default.

The next reviewed stage is:

1. live-qualify the explicit two-UE server path and freeze clock
   synchronization and maximum-lateness rules;
2. add velocity/time compensation and a validated uncertainty model;
3. compare the retained best-source rule against covariance-aware fusion on
   the same associations;
4. use the supervisor's 2-D geometry as an oracle/reference evaluator while
   replacing CARLA actor truth with associated model tracks at runtime; and
5. expose recipient-specific critical-object updates to the later downlink
   occluder reasoner.

The edge-terminal ACK and cooperative object update remain separate messages.
Task 3 may prioritize a compact object update without coupling it to transport
acknowledgement or returning dense segmentation over the radio.
