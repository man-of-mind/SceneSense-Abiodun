# Claude prompt — Phase-1 frozen-policy runtime corrective pass

Work in the existing repository and begin only after verifying that `HEAD` is
exactly `e1f600ac4a2794dbfd22700eb156d6e2c56b6f28`. This is a bounded corrective
pass over Claude's additive Phase-1 package. It is **not** authorization to
launch CARLA, OAI/RFsim, Docker, CUDA, the map server, a network service, a
training run, or a live policy pilot.

## Worktree safety

The worktree already contains user-owned and concurrent-agent changes. Record
the complete starting `git status --short`. Do not edit, stage, reset, delete,
rename, format, or otherwise absorb anything outside:

`rl_agent/splitfusion_hybrid_sac_live_route_b_v1/`

In particular, do not touch the OAI submodule, the technical report, any
Run-2/Run-3 diagnostic or presentation file, or any existing untracked prompt
or audit file. Do not modify frozen training artifacts or evidence under
`experiments/`. Do not push, pull, merge, rebase, or reset.

Before changing code, reproduce each issue below with the smallest CPU-only
test. Then repair it and retain the regression. Do not weaken an existing
gate or change a frozen Hybrid-SAC training contract to make a test pass.

## Required corrections

### A. Checkpoint and actor integrity

1. Recompute and verify the embedded `campaign_sha256` in
   `campaign_complete.json`; do not merely read its seed-17 report pointer.
2. Make loaded tensor state genuinely immutable from a caller's perspective.
   `MappingProxyType` around mutable tensors is insufficient. A caller must
   not be able to mutate the verified actor/critic tensors or cause a dynamic
   digest to bless post-verification mutation. Store a fixed verified digest,
   keep private cloned tensor ownership, and return defensive clones/read-only
   views where exposure is necessary.
3. Preserve Python, NumPy, and Torch global RNG state across **actor
   construction as well as inference**. Instantiating `nn.Linear` before
   loading weights currently consumes Torch RNG. Add a regression proving
   construction plus one decision is RNG-neutral.
4. Remove the unsafe pathname-hash/second-open gap around the optional
   `weights_only=False` full checkpoint. The live weight source remains the
   `weights_only=True` snapshot. If full-checkpoint corroboration is retained,
   consume the exact already-hashed byte sequence once (for example through a
   private in-memory stream), require its pinned digest before deserialization,
   and label the opt-in pickle boundary honestly. Do not claim that no pickled
   object is reconstructed when cross-verification does reconstruct one.

### B. State semantics and actuation stop gate

5. Do **not** fabricate a new episode-start proof for each repeated Route-B
   decision. An `EpisodeStartProofV1` authorizes one exact first
   decision/tensor/frame/time in one controller lineage. Phase 1 therefore
   qualifies only a first/genesis decision. Make that limitation explicit and
   fail closed on any attempted persistent/repeated controller use until a
   later sequentially trained policy exists.
6. The phrase `genesis_matched` is misleading when measured BSR or ages are
   nonzero. Rename/split it so it states exactly what is true: the 22-element
   previous-outcome block is held at genesis zero. Do not imply that the whole
   31-D vector lies in training support.
7. A support audit is not permission to actuate. Run-3 saw BSR and all four
   ages only at zero. Add an explicit fail-closed actuation gate: any nonzero
   live BSR/age may be recorded for telemetry analysis but must not be passed
   to the frozen actor as if its effect were learned. Do not silently zero,
   clamp, forward-fill, or drop the measured value.
8. Production construction must use the exact registered normalization and
   freshness policy. Move any override seam behind an unmistakable test-only
   API, or reject overrides on the production builder.

### C. Evidence and analyzer integrity

9. Make `expected_actor_state_sha256` mandatory for qualification analysis.
   Every opened-decision row—not merely one row somewhere in the file—must
   contain that exact digest, exactly 31 finite ordered features, a support
   audit, and a consistent execution identity. Missing/null/wrong-width data
   must fail, not be skipped.
10. Give session and terminal documents registered schemas and verified
    canonical self-hashes. On a completed run, verify terminal status, frame
    count, session binding, and frame-stream digest. An absent/incomplete
    terminal must never be reported as a qualified completion.
11. Preserve create-only behavior against directory replacement/symlink
    redirection. Retain a directory file descriptor and use relative
    `open(..., dir_fd=...)`/`O_NOFOLLOW` (or an equivalently strong approach)
    for later files. Add a test that replaces the pathname after writer
    construction and proves no evidence can be redirected.

### D. Scientific wording and reproducibility

12. Replace `DEPLOYMENT_VALIDATION` with an accurate label such as bounded
    live-testbed qualification; this is neither real-world deployment nor
    sequential online adaptation.
13. Correct BSR wording: with the registered scale, 0 bytes maps to 0, 1 byte
    maps to `log(2) ~= 0.693`, and integer values >=2 bytes clip to 1.0.
14. The reported 375-point Cartesian actor audit is diagnostic output
    variation, not policy validation. Either add a deterministic, tested,
    reproducible table/summary generator with exact inputs and digest, or
    remove the exact counts/range from the report. Do not describe it as
    reward, robustness, or learned-feature validation.

## Verification and stop gate

Run only CPU/read-only or temporary-directory tests. At minimum:

- the focused Phase-1 suite;
- the current controlled-audit module;
- the full `splitfusion_live_dispatch_v1` suite;
- the direct-edge-map and quality-feedback-probe suites;
- `py_compile` and `git diff --check`.

Assert that CUDA remains uninitialized and that no live process/service was
started. Commit only files under the permitted Phase-1 package in one clearly
labelled corrective commit. Stop after reporting:

- starting/final HEAD and exact changed-file inventory;
- each reproduced defect and its fix;
- focused and affected-suite results;
- any unresolved blocker;
- confirmation that all pre-existing dirty paths are untouched.

Do not begin UE telemetry-sidecar implementation, retraining, Phase 2 live
measurement, or policy actuation in this task.
