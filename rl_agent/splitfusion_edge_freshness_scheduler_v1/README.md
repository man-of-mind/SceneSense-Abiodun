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
sweep is 0, 5, 10, 20, 50, and 100 ms plus latest-only scheduling without a
fixed queue-wait budget.

## Terminal feedback contract

Every transmitted frame must receive exactly one terminal classification:

- `MAP_INSTALLED`;
- `SUPERSEDED_PENDING`;
- `SUPERSEDED_BEFORE_DECODE`;
- `SUPERSEDED_BEFORE_TAIL`;
- `SUPERSEDED_BEFORE_PUBLICATION`;
- `QUEUE_WAIT_BUDGET_EXCEEDED`;
- `PROCESSING_HORIZON_EXPIRED`;
- `OUT_OF_ORDER_ARRIVAL`;
- `TRANSPORT_INCOMPLETE`;
- `PROCESSING_FAILED`; or
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
not guarantee latency improvement. The first live candidate should use a
bounded pipeline with at most two frames in flight:

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

The completed 288-cell campaign remains the authoritative original runtime
surface. Counterfactual scheduling results must be labelled as simulation and
must not overwrite measured cell records.
