# Multi-UE cooperative-map ingress v1

Status: **filtering/association/server integration implemented; headless live action-50 demonstration ready for L10319**.

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

## Headless live two-UE/action-50 demonstration

`live_two_ue_action50_v1.py` is the technology demonstration for this stage.
Its primary self-contained mode creates and drives two lightweight CARLA egos,
owns the single simulation clock, attaches the registered
1280x720/120-degree RGB camera and 200k-pps/120-degree radar to each, executes
the exact locked action 50 (`AE64 / UINT4 / q=0.50`) through the resident
SplitFusion UE encoder and frozen edge tail, and delivers both compact
world-frame object updates to `MultiUESpatialMapService`. A passive mode for
two externally owned live vehicles remains available.

This deliberately reuses one resident localhost model stack for the two
logical UEs. It tests real CARLA sensors, real model detections, identity and
clock binding, filtering, Hungarian association and map updates. It does not
claim two-device networking or radio latency. In self-contained mode it
destroys only its own actors and restores the prior CARLA settings. In passive
mode it controls or destroys neither ego and changes no world setting.

The nominal service always receives unmodified model outputs. A separate,
fresh shadow service cycles through five deterministic faults: identical
duplicate, same-identity/different-payload conflict, non-finite coordinate,
one-second-old observation and +8 m XY bias. Faults can therefore never
contaminate the nominal map. No RGB, radar or mask arrays are retained; the
output consists only of compact JSONL observations, snapshots, fault outcomes,
a summary and a hash manifest.

The +8 m case is intentionally diagnostic rather than silently deleted. It
should fail the association gate against its true counterpart, but because it
is otherwise a valid finite detection, v1 may expose it as a separate
single-source track. Eliminating that false track requires the later M-of-N
tentative/confirmed track policy or a calibrated uncertainty/outlier model.

The exact L10319 procedure is in
`LIVE_TWO_UE_ACTION50_RUNBOOK.md`. A successful run must complete every fault
case and observe at least one real association containing both UE sources.

SciPy is required for `scipy.optimize.linear_sum_assignment`; both the system
Python and the registered CARLA virtual environment currently provide it.

## Deliberate boundary

This version does not estimate positional covariance, average positions,
perform JPDA, classify occlusion from learned tracks, or send downlink warnings.
The existing greedy confidence-weighted fusion remains historical evidence,
not an implicit default.

After this live demonstration, the next reviewed stages are:

1. freeze clock synchronization and maximum-lateness rules for deployment;
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
