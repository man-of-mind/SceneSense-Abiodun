# Run-4 physical queue calibration v1 — preregistration

**Status: implemented offline; not launched.** Live execution requires a new,
expiring one-use grant created after the implementation commit.

## Scope

This is a bounded physical queue-calibration campaign for constructing the
offline Run-4 environment. It is not policy validation, deployment evidence,
or a perception endorsement. It preserves the original capacity result as
`CAPACITY_QUALIFICATION_REFUSED` and consumes only the separately sealed robust
amendment at commit `a9477a6`.

The three byte loads are one exact AE64/UINT8 identity group:

| tier | action | payload/frame | offered at 10 Hz |
|---|---:|---:|---:|
| low | 40 | 126,237 B | 10.09896 Mb/s |
| medium | 39 | 374,264 B | 29.94112 Mb/s |
| high | 38 | 619,563 B | 49.56504 Mb/s |

They are `EMERGENCY_ONLY` catalogue entries. This byte-only experiment neither
qualifies nor recommends their perception behavior.

## Packetization

Every action uses 1,200 application bytes per full datagram. With the 24-byte
SSBURST header, 8-byte UDP header and 20-byte IPv4 header, a full IPv4 packet
is 1,252 bytes, below the registered 1,500-byte path MTU. Expected chunks per
frame are 106/312/517 (low/medium/high); the tail chunk is retained exactly.
The sender and receiver refuse packetization drift.

## Design and split

Two profiles (`FAVORABLE_STABLE`, `ADVERSE_STABLE`) × six payload orders =
**12 create-only cells**, each with three continuous 150-frame blocks at 10 Hz
(450 decisions/cell; 5,400 total). Execution order is shuffled once with seed
`2026092401`; partition membership never changes.

- FIT whole cells: L-M-H, M-H-L, H-L-M.
- VALIDATION whole cells: H-M-L, L-H-M, M-L-H.

The transition directions are disjoint. Validation therefore tests the reverse
directions, not an independent RF replication; a failure must be reported with
that limitation and cannot be rescued by pooling FAVORABLE into ADVERSE.

## Timeline

Receivers become ready first. The tagged sender then opens and binds its socket,
validates the complete block plan, and emits a create-only READY record without
sending traffic. Only then does the target-channel primer produce a fresh MCS
and prove its tiny queue drained. After that proof, the runner publishes one
create-only monotonic epoch contract 20 ms into the future, bound to the READY
record. The configured 20 ms lead is deliberately below the inherited 100 ms
primer-grant-to-first-decision maximum; the runner refuses the cell before
publication if the actual grant age leaves insufficient room. Both the armed
sender and profile replay schedule step 0 against that exact epoch. A late
sender, skipped actuator step, clamp, socket drop, freshness violation or epoch
mismatch fails the cell.

## Evidence and accounting

The sender emits its registered v1 summary and frame CSV. Exact accounting is
reconstructed from all 450 rows: action/tier/payload/chunks, feature bytes,
bytes handed to the UDP socket, socket handoffs and socket drops. Receiver losses remain measured
outcomes, but malformed packets, wrong chunk identity, foreign frame ranges or
ambiguous streams fail structurally.

Every declared T-tracer output must exist and contain at least one data row:
UE DCI grant, RLC buffer, PDCP ingress, RLC TX SDU/dequeue; gNB PUSCH power,
UL-MCS decision and PDCP delivery. Empty output is failure, never a nullable
zero.

## Authorization and cleanup

The operator grant binds a canonical UUID, six-hour maximum interval, exact
output path, current HEAD and parent HEAD, complete source-inventory digest,
config digest and amendment digest. Before any live mutation it is durably
consumed with `O_EXCL` plus file and directory `fsync`. A crash spends it; no
code path deletes/refunds the marker.

Every cell and the outer campaign have `finally` cleanup. RF restore, RAN
teardown, core teardown, zero residual tunnels/processes/containers, successful
trace extraction and a cold final host are mandatory. A failed cell is
preserved and never retried in place.
