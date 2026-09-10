# SplitFusion freshness-first edge scheduler v1

This package freezes the scientific contract for the next edge-runtime
optimization. It is a CPU-only prototype and does not alter the deployed,
hash-pinned SplitFusion runtime or the completed 288-cell evidence.

## Objective

The edge should maximize the utility of the **freshest map state**, not the
number of old frames it eventually completes. A newer complete frame may
therefore replace older pending work. Work already handed to CUDA is not
preempted; it is reconsidered at safe stage boundaries before decode, before
tail inference, and before publication.

The supervisor's 10 ms proposal is represented as an optional queue-wait
budget. It is not a default and must be tested prospectively. The planned
sweep is 0, 5, 10, 20, and 25 ms plus latest-only scheduling without a fixed
queue-wait budget. Values above 25 ms are excluded initially because the
measured path already has a large latency budget; they should be introduced
only if later optimization materially changes that budget.

## Terminal feedback contract

Every transmitted frame must receive exactly one terminal classification:

- `MAP_INSTALLED`;
- `SUPERSEDED_PENDING`;
- `SUPERSEDED_BEFORE_DECODE`;
- `SUPERSEDED_BEFORE_TAIL`;
- `SUPERSEDED_BEFORE_PUBLICATION`;
- `SUPERSEDED_PUBLICATION_PENDING`;
- `QUEUE_WAIT_BUDGET_EXCEEDED`;
- `PROCESSING_HORIZON_EXPIRED`;
- `OUT_OF_ORDER_ARRIVAL`;
- `TRANSPORT_INCOMPLETE`;
- `PROCESSING_FAILED`;
- `PUBLICATION_FAILED`;
- `PIPELINE_ABORTED`; or
- `IDENTITY_REJECTED`.

Supersession feedback carries the discarded frame identity, selected action,
capture and arrival timestamps, the replacing frame identity, discard stage,
age, queue wait, bytes already sent, and compute already spent. It is not an
installation ACK and must never be counted as one.

## Agent interpretation

An intentional freshness drop is not a radio failure. It earns no perception
installation utility, but the action is still charged for bytes and compute
already consumed. This avoids both incorrect extremes: punishing a deliberate
freshness decision as packet loss, or making an expensive stale action free.

Consequently, raw `installed / sent` remains a diagnostic rather than the
primary objective. The prospective analysis must report at least:

- fresh installations per captured and transmitted frame;
- time-weighted map AoI and time above the freshness limit;
- intentional supersession rate by stage;
- true transport and structural failure rates;
- bytes and compute spent on superseded work; and
- terminal-outcome reconciliation.

## Safe parallelism boundary

Arbitrary Python threads around one CUDA model are not safe preemption and do
not guarantee latency improvement. If frame 1 is executing while frames 2, 3,
and 4 arrive, the pending slot successively replaces 2 with 3 and 3 with 4.
After frame 1 reaches the next safe boundary, the worker takes frame 4 rather
than draining frames 2 and 3. The first live candidate should use a bounded
pipeline with at most two frames in flight:

1. socket receive/reassembly thread;
2. one GPU decode/tail worker, preserving model and CUDA-stream ownership;
3. one CPU publication/serialization worker where exact parity permits it;
4. depth-one latest-frame slots between stages; and
5. a supersession/deadline gate at every stage boundary.

GPU work for frame *t* may overlap CPU-only publication work for frame *t-1*.
Two concurrent calls into the same frozen model are not authorized by this
contract. The optimized tail remains subject to bit-identical tensor,
ordering, p025, segmentation, and serialized-record gates.

## Validation sequence

1. CPU queue, identity, terminal-accounting, and reward-attribution tests.
2. Offline 288-cell discrete-event replay with the measured pre-optimization
   and optimized service-time distributions.
3. Sweep the queue-wait hypothesis and select by map freshness/utility, not by
   installed count alone.
4. GPU parity and stage-overlap microbenchmark.
5. Short live CARLA/OAI diagnostic, initially actions 50 and 71, with 300--500
   prepared frames per action; expand to actions 30 and 15 only if needed to
   characterize the payload frontier.
6. Rebuild the RL transition data with explicit terminal outcomes before PPO
   training.

`run_sweep.py` implements the initial discrete-event screen over the qualified
four-action live timing diagnostic. It first reproduces the measured
single-worker/latest-pending behavior. The live evidence lacks timestamps for
frames displaced before result return, so those timestamps are reconstructed
by a frozen, within-action interpolation rule and remain labelled as imputed.
The reproduction gate must pass before the counterfactual policy rows may be
used. This first screen is not yet the full 288-cell environment simulator.

The completed 288-cell campaign remains the authoritative original runtime
surface. Counterfactual scheduling results must be labelled as simulation and
must not overwrite measured cell records.

## Two-stage execution candidate

`pipeline.py` implements the bounded concurrency primitive selected by the
offline sweep. It supports exactly two prospective policies:

- latest-only pending work with no fixed queue expiry; and
- latest-only pending work with a 25 ms pre-compute queue-wait budget.

An idle compute worker starts a frame immediately; 25 ms is an expiry ceiling,
not an intentional hold. While frame 1 is active, arrivals 2, 3 and 4 occupy a
single slot and successively supersede each other. Frame 1 is not interrupted;
the compute worker next receives frame 4. A second depth-one slot permits the
single CPU publication owner to overlap with the single compute owner. Pending
publication results may likewise be replaced, but an active publication is not
interrupted.

The primitive is bound to one run/cell/stream. It deliberately does not hide a
multi-UE fairness policy. Callback failures are structural and fail closed;
normal close drains both stages. Every offered frame receives exactly one
terminal outcome, and spent feature bytes, compute time, and publication time
remain chargeable even when a frame is intentionally superseded.

Pipeline feedback uses
`scenesense.splitfusion.edge_freshness_feedback.v2`; compared with the earlier
CPU contract it adds separately accounted publication time and publication-
queue terminal outcomes.

This is not yet a promoted live edge runtime. The current optimized tail keeps
a singleton snapshot between its tail and serializer calls, so splitting that
boundary requires a separate output-parity qualification. The deployed and
hash-pinned edge remains unchanged.

`runtime_bridge.py` binds this candidate to the existing SFD1 runtime without
weakening that boundary. The sole compute owner calls the complete frozen edge
operation and revalidates action, sequence, capture, stream, frame, and byte
identities. Only the already-serialized, identity-checked result reaches the
publication owner. This can overlap result publication/transmission with the
next edge call, but it does not yet claim that post-processing or serialization
itself has moved off the compute owner.
