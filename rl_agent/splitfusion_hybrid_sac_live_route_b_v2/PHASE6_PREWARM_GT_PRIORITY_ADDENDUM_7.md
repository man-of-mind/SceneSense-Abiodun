# Phase 6 addendum 7: deterministic pre-warm and reward-priority object GT

**Status:** `REGISTERED_BEFORE_ANY_NEW_PHASE6_EVIDENCE`, 2026-09-30. Base commit `4626c2e`.
This follows the offline audit's classification: **B — GT_PRODUCER_FIX_REQUIRED**, with pre-warm as a
prerequisite. Everything here is offline implementation and tests; nothing was executed live. The
records are `phase6_prewarm_gt_priority_addendum_7.json` and the prospective
`phase6_handshake_manifest_v7.json`. Together they bind all six prior attempts by hash.

The scientific contract is unchanged:
- the actor, the state, the reward and Q_perc;
- the 170-ms deadline and k_min = 2;
- telemetry, freshness and fallback;
- the hold protocol and latest-only edge scheduling;
- wire identities.

## Pre-warm

Every registered path is warmed before any frame exists. That is 12 joint modes, each at the lower,
middle and upper registered q_e4, for 36 executions. Each timed path is bracketed by
`torch.cuda.synchronize()`.

- **UE:** the real input preparation, front, ranker, encoders, quantizers and zstd, run before the
  runtime becomes ready.
- **Edge:** a synthetic SFD4 wire per path goes through the real processor: decode, tail,
  post-processing and map-update construction. READY is written only after every path succeeds.
- **Isolation:** synthetic inputs come from a local seeded generator. They never become frames,
  opportunities, tickets, map updates or reward records. The edge's frame counters and context session
  are isolated and restored.
- **Reports:** create-only, with per-path timings and payload/update hashes.

## Reward-priority object GT

- **Before the route:** `refresh_static(force=True)` is timed and must complete before any frame; a
  failure refuses admission.
- **Two bounded classes** replace the single FIFO:
  - HIGH: reward-requested policy frames;
  - LOW: holds and fallbacks, used only for report metrics.

  HIGH always goes before queued LOW, order is kept within each class, and LOW runs whenever HIGH is
  empty. The pinned worker and the object-GT computation are unchanged.
- **Bounded residual:** a LOW item already running is not pre-empted, so a HIGH item can wait for at
  most one LOW computation (about 90–120 ms measured).
- **Per-ticket instrumentation:** enqueue, worker start, queue wait, refresh, object-row construction,
  file write, object count, output hash/size/identity, completion. Semantic-GT instrumentation is
  unchanged.

## Next step (requires authorization)

One one-decision-plus-hold handshake, run as `phase6_handshake_manifest_v7.json` specifies and judged
by `handshake_verdict_v2`.
- If the reward frame is superseded after verified warm-up, the result is
  `REWARD_FRAME_PROTECTION_REQUIRED`, and the run stops.
- If feedback exceeds 170 ms, the verdict reports the exact excess by stage. The deadline and reward are
  not changed.
