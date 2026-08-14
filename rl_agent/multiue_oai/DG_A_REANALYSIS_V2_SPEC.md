# DG-A.1 corrected desk reanalysis v2

This specification repairs the original DG-A.1 allocation-contract bug without changing or overwriting the
completed DG-A experiment. The reanalysis reads the accepted run and writes a new sibling directory. It starts
no OAI, CARLA, RAN, core, sender, receiver, or tracer process.

## Frozen correction

- Time is discretized at the registered 50 ms controller tick.
- Per-UE demand arrivals use the production frame size (including chunk headers), deterministic phase credits,
  and the requested aggregate rho. Equal and hot/cold demand are staggered; synchronized demand shares one
  phase. The horizon is long enough to generate at least eight frames for the slowest UE in a cell.
- Each UE has one newest-pending slot. A newer arrival replaces an unsent frame. All arrivals remain in deadline
  denominators, including replaced and end-skipped frames.
- Decentralized hard-C1 uses one token bucket per UE at `0.70 * mu_N/N`. Centralized observable admission uses
  one aggregate token bucket at `0.70 * mu_N` and admits the oldest pending frame. Buckets hold at most one
  frame, matching the live action cadence. The provisional screen freezes the registered equal initial
  `c_hat=mu_N/N`; it does not invent unmeasured large-N estimator dynamics.
- Admitted frames enter per-UE FIFO queues. A work-conserving fluid max-min server shares measured cell service
  equally across currently backlogged UE queues. Completion latency is measured from demand arrival, not from
  admission. Payload serialization and queueing therefore determine the 0.25/0.50 s deadline results.
- Two allocation envelopes are always emitted: ideal max-min and measured-residual max-min. The latter scales
  service by A4's clipped heavy-residual fraction. They remain separate provenance rows even if both equal 1.0.
- The static max-min allocation is also emitted as an audit field. Under N=50 hot-20% demand it must serve cold
  UEs fully before sharing residual capacity among hot UEs; uniform proportional demand scaling is forbidden.

## Decision lock

The original measured N=2 pair decision is recomputed unchanged. A provisional N=50 candidate gap exists only
when the registered meaningful-effect rule and the additional Pareto review both pass for every restart block,
all three registered service families, and both allocation envelopes for the same `(N, demand, payload)`.
The Pareto review permits no deadline regression and at most 5% latency, starvation, or goodput regression.
Pareto safety is an explicitly post-registration conservative review condition, not presented as part of the
original gate.

The result is still model-based. A positive result authorizes at most human review of an N=4 attach-only smoke;
it does not authorize DG-B automatically, the identification campaign, a controller ladder, or RL. A negative
result is the pre-registered cheap NO. The source experiment directory remains the immutable audit record of
the original analyzer and its superseded candidate verdict.

## Required sibling artifacts

- `source_provenance.json` with source/config hashes;
- `resolved_model_config.yaml`;
- `results_summary.json` and `DG_A_REANALYSIS_DECISION.md`;
- `large_n_sensitivity.csv` with both controllers' queue/deadline metrics;
- `artifact_manifest.json` and a terminal `COMPLETED.json`.

