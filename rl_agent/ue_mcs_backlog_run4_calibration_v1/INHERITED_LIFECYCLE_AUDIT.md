# Inherited lifecycle audit

This package is additive. It does not copy or silently fork the proven OAI
lifecycle. `Runner` inherits from
`ue_mcs_backlog_near_capacity_v1.capacity_runner.Runner`, and runtime
`inherited_lifecycle_audit()` refuses unless every inherited method below still
comes from its registered owner and every owner file matches its SHA-256 pin.

## Capacity-runner lifecycle retained

- bounded subprocess execution and timeout lookup;
- cold-RAN/tunnel checks;
- hash-bound 273-PRB/4D5U launcher;
- bounded RAN teardown (INT, TERM, then recorded KILL failure);
- bounded T-tracer extraction;
- Docker core state and immutable image inspection;
- ext-DN context binding and radio-route proof;
- core teardown.

## Scientific-runner probes retained

- real UDP path probe requiring ext-DN arrival and UE PDCP evidence;
- target-channel primer requiring a fresh UE-decoded round-0 UL grant;
- primer queue-drain proof and first-decision ordering check.

## Base primitives retained

- managed child-process bookkeeping;
- UE/gNB tracer startup;
- telnet channel actuation/read-back;
- RF restoration;
- traffic process completion.

## Deliberately replaced

- constructor: the old constructor would consume the refused capacity result;
- traffic launch: uses 1,200-byte application chunks, arms the sender before
  the primer, and publishes a READY-bound shared future epoch only after drain;
- cell loop: channel replay and the already-armed sender use the same monotonic
  epoch while retaining the inherited 100-ms primer freshness gate;
- traffic audit: consumes the real sender schema, with exact per-frame byte and
  datagram reconciliation;
- T-tracer gate: every declared required event needs at least one evidence row;
- final cold check: also searches for the new sender module;
- plan/manifest/seals: bind the sealed robust amendment and whole-cell
  FIT/VALIDATION partitions.

No inherited analyzer is used. Analysis begins only after this capture package
has produced a sealed, structurally valid campaign.
