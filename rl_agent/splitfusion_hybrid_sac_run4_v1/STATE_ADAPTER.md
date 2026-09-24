# Run-4 causal UE state adapter

`state_adapter.py` is the narrow pre-calibration bridge from raw UE trace
records to the typed Run-4 contract. It does not read trace files, cache
measurements, normalize features, or choose an action.

For UL MCS, it accepts only the latest UE-decoded grant whose source and
availability timestamps are strictly before the state commit and whose
identity matches the decision session and UE. The grant must be uplink,
table 0, HARQ round 0, and bound to `SCENESENSE_MCS_POLICY=sinr`.
Retransmissions, future records, other UEs/sessions, other clock domains and
foreign scheduler policies are ineligible. NDI may be either zero or one:
new data is the NDI toggle represented by HARQ round 0, not the literal value
one.

For RLC backlog, it accepts only the latest matching UE sample whose source
and availability timestamps are strictly before both the payload-enqueue and
state-commit boundaries. A measured empty queue remains valid `0`; absence is
an invalid `ScalarObservationV1` with `value=None`.

Both selectors are stateless. Therefore an empty input on a later decision
cannot inherit an earlier value. Freshness thresholds, empirical scaling and
the conservative fallback action remain external decisions to be bound after
the 12-cell calibration.
