# Claude prompt — UE-state evidence audit before Hybrid-SAC retraining

You are implementing a bounded, read-only UE-state evidence audit before the
next SplitFusion Hybrid-SAC retraining. This phase must determine whether
existing OAI traces can support a causal, action-conditioned BSR feature. It
must not modify the policy, environment, reward, runtime, or training code.

## Scientific objective

Establish from code and retained evidence, not assumptions:

1. Which UE-side source can provide backlog observed immediately before an RL
   decision and before the current SplitFusion payload is enqueued.
2. Whether `NRUE_MAC_RLC_BUFFER_STATUS.csv` or
   `NRUE_MAC_BSR_STATUS.csv` is the correct source for that observation.
3. Whether retained traces can be aligned within the same run to UE grant/DCI,
   MCS, gNB PUSCH/SNR, offered traffic/enqueue, and transmitted-byte evidence.
4. Whether the evidence identifies

   ```text
   (previous backlog, SplitFusion action/payload, radio service)
       -> (next backlog, delay/failure)
   ```

   or merely characterizes BSR under a fixed traffic generator.
5. What is safe to use in retraining now and what needs a small future
   calibration run.

Do not assume zero BSR means zero network delay. A valid pre-action zero means
no older reportable UE backlog at that measured layer and instant. It does not
prove there are no packets in flight, HARQ activity, scheduler delay,
gNB/core/edge queuing, or delay from the payload about to be enqueued.

## Hard scope boundary

This phase is offline and read-only with respect to experiment evidence.

Do not:

- launch CARLA, OAI, RFsim, Docker, the map server, CUDA, or model inference;
- initialize a CUDA context;
- launch or resume RL training;
- alter existing experiment evidence;
- modify the Hybrid-SAC state, reward, environment, trainer, replay buffer,
  live runtime, or frozen contracts;
- create a guessed BSR model or inject random BSR values;
- select or freeze a new normalization constant;
- claim UE visibility for gNB-side PUSCH/SNR evidence;
- join timestamps from different clock domains without a measured bridge;
- run a new live calibration in this phase.

This is an evidence audit only.

## Worktree safety gate

Before doing work:

1. Read and obey the repository's `CLAUDE.md`.
2. Record starting HEAD, branch, full `git status --short`, and hashes of
   existing user-owned dirty files.
3. Preserve every pre-existing dirty and untracked path exactly.
4. Do not use `git reset`, `git clean`, `git stash`, checkout restoration,
   pull, merge, rebase, or push.
5. Work only in this new directory:

   ```text
   rl_agent/splitfusion_hybrid_sac_ue_state_audit_v1/
   ```

If that directory already exists, HEAD changes unexpectedly, or another
process changes one of your files, stop and report the conflict.

## Exact deliverables

Create exactly these three additive files:

```text
rl_agent/splitfusion_hybrid_sac_ue_state_audit_v1/
├── audit_ue_state_evidence.py
├── test_audit_ue_state_evidence.py
└── UE_STATE_EVIDENCE_AUDIT.md
```

Do not edit `__init__.py` or any existing file. The audit module must be
import-side-effect free and must not read evidence until an explicit function
or CLI entry point is invoked.

## Evidence discovery and provenance

Begin with retained UE traces under paths matching:

```text
rl_agent/experiments/**/ttracer/ue/csv/NRUE_MAC_BSR_STATUS.csv
rl_agent/experiments/**/ttracer/ue/csv/NRUE_MAC_RLC_BUFFER_STATUS.csv
```

For every logical run used, discover same-run companions where present,
including `NRUE_MAC_DCI_GRANT.csv`, `live_mcs.csv`,
`live_pusch_with_ingest.csv`, traffic/send/enqueue evidence, and throughput or
transmitted-byte evidence.

Do not join files merely because their names look related. Define a logical
run identity from the directory structure and refuse cross-run joins.

For every source used, record:

- relative path, byte size, SHA-256, row count, and exact header/schema;
- timestamp fields and apparent clock domain;
- missingness and sampling cadence;
- source-side visibility: UE, gNB, traffic generator, or unresolved.

Inspect the logger/source implementation that emits both BSR-related CSVs.
Cite exact local file paths and line numbers in the report. Determine where in
the MAC/RLC processing sequence each row is emitted. If code cannot prove
whether a sample precedes or follows enqueue/multiplexing, classify causality
as unresolved.

## Required BSR comparison

Compare `NRUE_MAC_RLC_BUFFER_STATUS` and `NRUE_MAC_BSR_STATUS` explicitly.
For each, answer:

- What object is measured?
- Are byte fields queue occupancy, encoded BSR indices, reported values, or
  post-multiplex residual state?
- Are all logical channel groups represented, and how is total backlog
  derived?
- At what event is the row emitted?
- Can it be sampled at the RL decision point before current-payload enqueue?
- What precisely does zero mean?
- Is it directly available inside the UE runtime?
- Could it include the current action payload and leak the action into its own
  state?

A source qualifies as pre-action BSR only if evidence proves:

```text
read older UE backlog
        ↓
construct policy state
        ↓
select (mode, q)
        ↓
construct/enqueue current payload
```

If neither retained source proves this ordering, report
`PRE_ACTION_BSR_SOURCE_UNRESOLVED`. Do not choose by intuition.

## Same-run alignment audit

Within each run, assess alignment of BSR/RLC data with grants, MCS, PUSCH/SNR,
and traffic. For every proposed alignment:

- state join keys and whether clocks are identical, bridged, or unresolved;
- state and justify maximum permitted skew;
- report matched-row coverage and skew quantiles;
- reject ambiguous many-to-many joins;
- preserve missing values as missing;
- distinguish UE-visible DCI/MCS from gNB-observed PUSCH/SNR.

A nearest-time join is not evidence unless files share a clock domain and the
allowed skew follows from measured cadence.

Where fields permit, report per-run (not only pooled):

- total-byte zero fraction, P50, P90, P95, P99, and maximum;
- sampling interval P50/P95/P99;
- MCS and grant distributions;
- alignment coverage;
- lagged descriptive association between offered bytes, granted/transmitted
  bytes, and subsequent backlog.

Treat correlations as descriptive, not causal proof.

## Action-conditioned coverage decision

Determine whether retained runs vary SplitFusion actions or payload sizes
while measuring backlog evolution. Assign exactly one final status:

- `ADEQUATE_FOR_ACTION_CONDITIONED_QUEUE_MODEL`
- `ADEQUATE_FOR_CARRIER_AND_NORMALIZATION_AUDIT_ONLY`
- `INSUFFICIENT_OR_CAUSALLY_UNRESOLVED`

Fixed-rate traffic is not enough to identify how 12 modes and continuous `q`
affect future backlog. If retained traffic is fixed or narrower than the
SplitFusion payload range, say so and do not extrapolate.

If more evidence is required, specify but do not run the smallest calibration:
a few registered SplitFusion actions spanning small, medium, and large
payloads under at least contrasting radio conditions, timestamping policy
state sampling, payload enqueue, pre-enqueue BSR/RLC, grants/transmitted bytes,
next backlog, and feature completion or failure. Do not prescribe a new
288-cell campaign unless evidence proves a smaller calibration cannot answer
the question.

## Normalization analysis

Evaluate the current transformation:

```text
clip(log1p(bsr_bytes), 0, 1)
```

Quantify positive-value saturation and explain why it collapses nearly all
nonzero queues. Provide candidate diagnostics only, such as scales based on a
train-only P95/P99 or registered engineering bound. Show zero preservation and
clipping percentages per run. Label every candidate
`PROVISIONAL_NOT_FROZEN`.

Do not select a final scaler until the causal source, split, and SplitFusion
payload coverage are established.

## Required report conclusions

`UE_STATE_EVIDENCE_AUDIT.md` must contain:

1. Executive decision.
2. Exact meaning of zero BSR.
3. Side-by-side RLC-buffer versus MAC-BSR comparison.
4. Source-code provenance and timing order.
5. Same-run alignment table.
6. Per-run and pooled distribution tables.
7. UE-visible versus gNB-only metric classification.
8. Action-conditioned coverage verdict.
9. Normalization saturation analysis.
10. Retraining implication: whether BSR may enter learned state now, whether it
    remains an external guard pending calibration, and which fields must not
    be fabricated.
11. Minimal future calibration design, if required.
12. Limitations and explicit non-claims.
13. Complete artifact/hash inventory.
14. Worktree and execution-safety statement.

Do not claim existing BSR evidence improves the policy unless established. Do
not imply SNR, MCS, and BSR are interchangeable; explain their complementary
meanings.

## Tests

Use `unittest`, CPU-only, primarily temporary synthetic fixtures. At minimum
cover:

- parsing both BSR schemas and exact LCG/RLC total calculation;
- zero and missing-value preservation;
- rejection of negative or malformed byte counts;
- logical-run isolation and refusal of cross-run joins;
- rejection of unresolved clock-domain joins;
- one-to-one alignment and skew accounting;
- explicit pre-enqueue versus post-enqueue classification;
- prevention of action leakage;
- normalization saturation calculations;
- per-run statistics without accidental pooled weighting;
- deterministic results and SHA-256 evidence binding;
- import causing no filesystem scan, subprocess, network, CUDA, Docker,
  CARLA, or OAI activity.

A bounded real-evidence smoke test may verify file counts and schemas, but it
must not rewrite, copy, truncate, or normalize evidence in place.

Run:

```text
python3 -m py_compile <the two Python files>
python3 -m unittest <the focused test module>
git diff --check
```

Also verify CUDA remains uninitialized and no prohibited runtime was launched.

## Completion and stop gate

Review the report against raw evidence before committing. Commit only the
three permitted files in one concise commit. Do not stage any pre-existing
dirty file.

At completion, report starting/final HEAD, commit hash, files, test counts,
final evidence status, recommended pre-action BSR source or `UNRESOLVED`,
whether evidence supports action-conditioned SplitFusion queue dynamics,
whether bounded calibration is required, surprising findings, runtime-safety
confirmation, and confirmation that user-owned dirty paths stayed untouched.

Then stop. Do not change the environment or begin retraining.
