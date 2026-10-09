# Run-4 physical queue calibration v2 — prospective preregistration

**Status: offline implementation under review; never launched.** A v1 grant
cannot authorize this design. Live work requires a new v2 one-use grant after
the final reviewed commit.

## Why v2 replaces the unexecuted v1 tier plan

The v1 tiers covered 126,237–619,563 bytes, while the frozen Run-4 modeled
support is 6,423–427,605 production `total_transmitted_bytes`. That omitted the
small, high-compression region needed by warm-up and would make the fitted
queue model refuse valid Run-4 actions. V2 prospectively replaces those tiers;
no evidence was collected under either plan.

The v2 bytes are derived from the exact 391 train-scene production payload
surface, never from the 72-action presentation median. Low is the
inverse-inclusion-weighted 25th percentile of action 71 and lies just below the
support floor. Medium is the weighted median of action 22 and lies between the
historical adverse/favorable capacity estimates. High is the weighted median
of action 38 and exceeds the support ceiling and both estimates.

| tier | byte-only source identity | fixed bytes/frame | 10-Hz application load |
|---|---|---:|---:|
| low | action 71, AE32/UINT4, q=.98 | 6,398 | 0.51184 Mb/s |
| medium | action 22, AE128/UINT8, q=.90 | 248,703 | 19.89624 Mb/s |
| high | action 38, AE64/UINT8, q=.50 | 619,858 | 49.58864 Mb/s |

These identities only authenticate observed byte coordinates. They do not
recommend or qualify the actions' perception behavior.

## Packet and d=2 identity

The calibration replays the numeric production byte coordinate using separate
1,200-byte application chunks plus a 24-byte SSBURST header. This is MTU-safe
calibration packetization and is explicitly not production packetization.

The 450 tensor frames in a cell remain at 10 Hz. Consecutive pairs form 225
causal policy decisions: tensor `2t` is `REWARD_REQUESTED`; tensor `2t+1` is
`HELD_NO_REWARD` with held ordinal 1. Each 150-frame block therefore contains
75 complete d=2 decisions and never splits a pair. Sender and receiver audits
must preserve every frame/chunk identity and produce first-arrival,
last-arrival, completion, and missing-frame endpoints. This calibration does
not itself implement reward, latest-only tail service, or next-state coupling;
it only provides the physical transport/queue evidence those later components
may consume.

## Design

Two stable channel profiles × six whole-cell load orders remain 12 cells.
Each cell has three 150-frame blocks. FIT uses the three cyclic orders and
VALIDATION their reverses, so validation tests direction extrapolation rather
than an independent RF replication. A result must retain that limitation.

Every RF command, source identity, exact 450-row trace prefix, sender event,
receiver event, T-tracer schema/identity, and teardown state is fail-closed.
Zero received streams is retained as measured loss; malformed or ambiguous
identity is not.
