# Run-4 live-qualification package v2 — Phase 4: action hold and 170-ms reward ticket

**Verdict:** `PHASE4_OFFLINE_HOLD_TICKET_AND_COMPOSED_PIPELINE_PASSED`. Offline, CPU only.

## Implementation

`reward_hold_controller_v2.py` is a single serialized event loop. Worker threads only `post()`;
all state changes happen in the loop's `run_pending()`.

- **Deadline.** `REWARD_DEADLINE_NS = run4_contract.REWARD_DEADLINE_NS = 170,000,000`,
  inclusive, measured from action open to UE feedback receipt. This is the Run-4 share of the
  ~200-ms capture-to-feedback budget. The obsolete Run-3 `reward_ticket_controller` (200 ms) is
  neither edited nor reused.
- **Timeout.** A timeout resolves at `action_open + TIMEOUT_RESOLUTION_ELAPSED_NS` (170,000,001)
  through `resolve_reward`, with reward −1.
- **Hold.** `K_MIN = MINIMUM_HOLD_TENSORS = 2`. The first tensor of an action carries
  `reward_requested = true`. Every later tensor reuses the exact action (mode, `q_e4`, bundle,
  nullable anchor) with `reward_requested = false`. A new decision is allowed only when the
  ticket is resolved **and** at least two tensors have been sent. An unresolved ticket keeps
  holding.
- **Feedback identity.** Feedback must match session, controller lineage, decision, ticket,
  reward frame, tensor, capture timestamp, mode, exact `q_e4`, execution bundle, nullable
  anchor `action_id`, and `reward_requested = true`. The UE-local receipt time is stamped by
  the loop.
  - A byte-identical duplicate is `DUPLICATE_IGNORED`.
  - A conflicting duplicate, or an identity mismatch, raises `ConflictingFeedbackError` and
    faults the controller.
  - Feedback after the boundary is `LATE_ORPHAN`; feedback for an unknown request is
    `UNKNOWN_ORPHAN`. Neither can close a newer ticket.
- **Faults.** Infrastructure and evaluator faults are excluded (`learning_included = False`,
  no reward). Because the contract forbids them as a previous outcome, the next decision
  requires a new session (`SessionBreakRequired`). A timeout alone does **not** terminate.
- **Previous outcome.** It comes only from `PreviousOutcomeV1.from_resolution`.
- **Map path.** Map ACKs go to a separate ledger and never touch reward state.

`live_state_v2.py` (no telemetry parsing) builds the exact 21-D state:

1. `guard_state_for_action` with the exact training freshness (`6c694ebe…`);
2. the registered raw-support refusal from `transport_model_v2.json` (MCS [8, 28]; backlog
   [0, 36,002,536] B), checked before any clip;
3. `build_policy_features` with the exact training scaling, reconstructed from the registered
   FIT catalogue. Its digest `cb1a3e4d…5671` equals the checkpoint's
   `empirical_scaling_sha256`.

## Tests (`test_phase4_reward_hold_v2`: 14 OK)

- An ACK at exactly 170 ms is SUCCESS (latency 170.0, reward q−0.25). One nanosecond later it
  is TIMEOUT −1, and the ACK becomes `LATE_ORPHAN`.
- A lost decision tensor or a lost ACK times out. Decisions continue after a timeout.
- Minimum two tensors: early feedback does not permit a decision until the second tensor.
  Held frames never request reward and reuse the exact action. An unresolved ticket blocks new
  decisions.
- The prior outcome enters the next state exactly (one-hot, `prev_q`, `q_perc`, latency/170,
  present/success).
- Duplicates are ignored. Conflicting feedback, and every identity-field mismatch, fail closed.
- Late and unknown orphans never close a newer ticket. An evaluator fault breaks the session,
  not the policy. Map ACKs are independent.
- The live state refuses MCS 7 and a backlog of 36,002,537 B before the actor. The training
  bindings are exact.
- **Composed pipeline.** Typed synthetic state → the frozen seed-43 actor (CPU) → dynamic
  identity seam → SFD3 UE/edge (fake codec) → independent map and evaluation **worker
  threads** → ACK → ticket closure. The script had 8 decisions with feedback at
  60/90/170/lost/171/40/lost-tensor/120 ms, plus a duplicate of every ACK. Results:
  - terminals are exactly SUCCESS, SUCCESS, SUCCESS, TIMEOUT, TIMEOUT, SUCCESS, TIMEOUT, SUCCESS;
  - 8 actor calls produced 8 resolutions;
  - ACCEPTED count equals the SUCCESS count, and there are exactly 2 late orphans;
  - every reward frame is its own decision's first tensor, and every held frame carries the
    identical action;
  - every delivered frame, held frames included, was installed and map-ACKed (all but the
    one scripted lost tensor);
  - evaluation saw only the 7 delivered reward frames;
  - every decision had at least 2 tensors;
  - no deadlock (bounded queue waits, workers joined), and CUDA was never initialized.

Package total, run together: 112 tests OK (Phases 1–4 plus `test_run4_contract`).

## Files

Created: `reward_hold_controller_v2.py`, `live_state_v2.py`, `test_phase4_reward_hold_v2.py`,
`PHASE4_REPORT.md`. Nothing outside this package is modified. No telemetry parsing or clock
estimation is in Phases 3–4.
