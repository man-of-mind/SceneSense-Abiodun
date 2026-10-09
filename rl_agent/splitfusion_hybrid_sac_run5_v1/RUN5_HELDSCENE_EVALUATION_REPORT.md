# Run-5 held-scene evaluation v1: results

**Scope.** This is an independent **modeled** held-scene validation: 255 unseen
`held_scene` scenes × 4 profiles × 3 validation seeds, run through the registered
transport model and joint channel. It is not live-system performance.

**Run.** The sealed sweep ran once at commit `ab96756`, with manifest
`b6d8b1aa…2c45`. Outputs are off Git and byte-unchanged, under
`evaluation_runs/heldscene_eval_v1/`:

| File | SHA-256 |
|---|---|
| `OUTPUT_MANIFEST.json` | `cba1a71b…7825` |
| `SUMMARY.json` | `7ce27b87…4d41` |
| `EVALUATION_COMPLETE.json` | `7e656dd3…e0c2` |

Hashes and headline numbers are bound in `RUN5_HELDSCENE_EVALUATION_EVIDENCE.json`.

**Completeness.**
- 116,280 decision rows (12 contexts × 38 policies × 255 decisions) and
  8,372,160 restricted-oracle anchor scores.
- 0 faults.
- An identical exogenous tape across all policies in each context.
- 23–24 of the 72 anchors were refused at every decision as outside the transport
  support.
- There were no queue-support refusals.

**Update-10,000 results, pooled over the 12 contexts** (reward includes timeouts
at −1):

| Policy | Reward | Success | Q_perc (executed) | Latency P50/P95/P99 (ms) | Censored | One-step oracle regret |
|---|---|---|---|---|---|---|
| Run-5 seed 17 | 0.2054 | 0.848 | 0.557 | 86.2/126.2/154.9 | 464 | 0.122 |
| Run-5 seed 29 | 0.1932 | 0.845 | 0.558 | 92.4/130.6/156.1 | 474 | 0.136 |
| Run-5 seed 43 | 0.2071 | 0.847 | 0.566 | 88.8/129.7/156.5 | 467 | 0.121 |
| Run-5 mean [min, max] | 0.2019 [0.1932, 0.2071] | 0.847 | 0.561 | 89.1/128.8/155.9 | 468 | 0.126 |
| Run-4 seed 43 | 0.2035 | 0.848 | 0.555 | 86.8/126.6/154.8 | 464 | 0.125 |
| Fixed action (mode 11, q 3000) | **0.2239** | 0.850 | 0.573 | 83.3/123.7/156.2 | 458 | 0.104 |

**Paired comparisons.** Run-5 minus Run-4 is +0.002, −0.010 and +0.004 for seeds
17, 29 and 43: mixed in sign and small. Run-5 minus the fixed action is −0.018,
−0.031 and −0.017, negative in all 12 contexts for every seed.

**Shuffled SNR.** It changed the argmax mode on 8–22% of decisions, while the
closed-loop reward changed by at most 0.0011.

**Checkpoints** (a learning diagnostic, not a selection): mean reward rises from
0.162 at update 500 to 0.202 at update 10,000, and regret falls from 0.166 to
0.126. Update 10,000 remains the only registered actor.
