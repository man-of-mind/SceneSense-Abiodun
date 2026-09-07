# Multi-UE cooperative-map ingress v1

Status: **initial implementation; no live or fusion qualification**.

This package is the first production-facing Task-2 layer. It validates compact
world-frame object updates from multiple UE streams, retains a bounded history,
and selects one causal source frame per UE/session/stream for a requested map
time. Every snapshot reports its cross-source capture-time spread and whether
it satisfies the caller's alignment tolerance.

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

## Deliberate boundary

This version does **not** associate detections, average positions, create
tracks, classify occlusion, or send downlink warnings. In particular, a model
confidence score is not a calibrated positional covariance and is not used as
one. The existing greedy fusion remains historical evidence, not an implicit
default.

The next reviewed stage is:

1. freeze clock synchronization and maximum-lateness rules;
2. add velocity/time compensation and class-aware gated assignment;
3. compare best-source, arithmetic mean, and covariance-aware fusion on the
   same associations;
4. maintain per-track source provenance, measurement timestamp, AoI and
   uncertainty;
5. use the supervisor's 2-D geometry as an oracle/reference evaluator while
   replacing CARLA actor truth with associated model tracks at runtime; and
6. expose recipient-specific critical-object updates to the later downlink
   occluder reasoner.

The edge-terminal ACK and cooperative object update remain separate messages.
Task 3 may prioritize a compact object update without coupling it to transport
acknowledgement or returning dense segmentation over the radio.
