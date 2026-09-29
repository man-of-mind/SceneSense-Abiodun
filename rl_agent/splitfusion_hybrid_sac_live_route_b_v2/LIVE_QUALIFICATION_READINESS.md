# Run-4 live qualification — Phase-5 readiness plan (plan only)

**Status:** `COMPONENTS_QUALIFIED__PHASE6_RUNNER_NOT_YET_BUILT`. Nothing here was launched. Phase 6
(the 300-frame CARLA/FCOS live qualification) needs its own explicit authorization. It also
cannot run until the remaining item in §4 is built.

## 1. What is proven, and where

| Phase | Result | Commit | Evidence |
|---|---|---|---|
| 1 Actor | seed 43 / update 10,000 restored bit-identically; CPU weights-only export; boundary `b61f27a9…bd3`, weights `d064013d…8e29` | `1900703` | `PHASE1_REPORT.md` |
| 2 Telemetry | bounded live provider; one radio-only run passes 10/10 pre-registered gates (100% fresh coverage, 300/300 raw-replay agreement) | `557662f`, `092f6a4` | `PHASE2_TELEMETRY_REPORT.md` |
| 3 Execution | continuous `(mode_id, q_e4)` identity seam equals the training path for all 117,612 pairs; SFD3 UE/edge parity | `3e12847` | `PHASE3_REPORT.md` |
| 4 Hold/ticket | 170-ms inclusive deadline, k_min = 2, orphans/duplicates/faults; composed pipeline from state through the frozen actor to ticket closure | `19efb01` | `PHASE4_REPORT.md` |
| 5 Plan | sealed binding manifest and proposed plan | this commit | this file |

The Phase-2 blocker was withdrawn in `PHASE2_BLOCKER_CORRECTION.md`. The live T-tracer transport
already existed.

## 2. Sealed binding

`LIVE_BINDING_MANIFEST_V2.json` (`manifest_sha256 3bda9494f3138a6a3639a8fcaf536a27ee74a3cede7ddb10c420f42cc9a686af`)
binds:

- **Actor:** seed, update, boundary, weights, export manifest, and the five pinned Run-4
  sources.
- **Contracts:** Run-4 schema/feature/reward/transition digests, the 21-feature order,
  action identity, the action catalog, q support, training freshness (`6c694ebe…`), training
  scaling (`cb1a3e4d…`), radio support, the 170-ms deadline, the timeout resolution, and
  k_min = 2.
- **Telemetry:** provider and qualification-runner source digests, gates digest, and the
  qualification `TERMINAL`/`RESULT` digests (status `PASSED`).
- **Execution:** the dynamic-execution runtime binding, the behavioral-source binding, and the
  perception/ranker/AE128/AE64/AE32 checkpoints, each hash-verified.
- **Radio:** every `radio_binding` pin as observed (0 problems) and the emitter pins. The
  uncommitted OAI working tree is labelled as not authored by this task.
- **Package:** the digest of every v2 runtime source.

`python -m rl_agent.splitfusion_hybrid_sac_live_route_b_v2.readiness_v2 verify` rebuilds the
manifest from the working tree and fails on any drift. It is the first preflight step.

## 3. Proposed Phase-6 plan

`live_qualification_300_v2.json` holds the machine-readable plan:

- run parameters: 300 frames at 10 Hz on Route-B, `FAVORABLE_STABLE`, CARLA `Epic`;
- CPU batch-1 actor, 5-ms decision lead, k_min = 2, 170-ms deadline;
- ordered preflight, launch sequence and teardown;
- evidence fields and gates P1–P6, with reward, success and latency reported but **not**
  gated.

Preflight binds the CUDA/model preload gates: every checkpoint is hash-verified before load,
there is one warm-up per module, and no hot-path loads are allowed. The direct edge-to-map
endpoint (`192.168.70.129:39320`, resolved from the CN5G bridge) and a distinct UE
reward-feedback endpoint are bound too. Teardown is restore read-back equal to the initial
state, followed by a host application-cold check.

## 4. Remaining before Phase 6 can be authorized

1. **Phase-6 live runner (not built).** The v2 seams are proven offline and, for telemetry,
   on the radio. No runner yet composes them with the live stack:
   - CARLA sensor preparation feeding camera SI / radar P40 into `live_state_v2.scene_observations`;
   - the real zstd/UINT codecs and CUDA front/tail inside the Phase-3 UE/edge runtimes;
   - the direct edge-to-map publisher;
   - a per-frame quality evaluator producing `RewardFeedbackV2` for reward-requested frames,
     sent back over the radio to the UE controller.

   So `proposed_command` in the plan is explicitly not runnable. Building this runner is
   integration work; it needs no new measurement framework.
2. **External fallback action.** When the guard refuses, the frame must still be sent. The
   exact fallback `(mode_id, q_e4)` and its logging must be pinned in the runner before
   authorization. It is not defined by any Run-4 contract I found, so it needs an
   Abiodun–Codex decision.
3. **Known live risks, reported rather than gated:**
   - MCS decision age was 50–78 ms, 21.5 ms inside the 100-ms bound; CARLA-driven payload
     timing may push some decisions into fallback;
   - backlog was 0 at every radio-qualification decision;
   - NDI-0 and retransmission-round rules are tested offline only;
   - the UE-feedback ACK latency is host-local, which makes it a lower bound.
4. **Not claimed:** CUDA/model-output equivalence of off-anchor actions, perception quality,
   reward or live performance.

## 5. Verdict

The Phase 2–4 components are qualified. The package is **not yet ready to launch** a Phase-6
live qualification: the runner in §4.1 and the fallback decision in §4.2 are open.
