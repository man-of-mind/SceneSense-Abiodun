# SplitFusion FCOS V1 numerical recovery

State: `UNQUALIFIED_IMPLEMENTATION_ONLY`.

Prospective amendment: `PROSPECTIVE_EPOCH10_GATE_ONLY`. The retrospective qualification attempt failed and makes no qualification or qualified-to-train claim. Its Replay A/B numerical-agreement gate is retired because the two fixed CUDA trajectories are nondeterministic. No additional replay, tau sweep, or disposable epoch is authorized.

The prospective gate verifies the fixed Replay A and Replay B hashes, fixes the preregistered yaw candidate at `tau=1e-2`, and constructs the breaker ceilings as ten times the maximum over the union of both 446-update healthy histories. The fixed threshold affects zero of the 831 update-447 yaw carriers in either replay. With an authorization bound to the exact committed source and evidence, `epoch10_gate.py` permits one create-only scientific epoch 10 from the verified original epoch-9 checkpoint. It writes a full atomic epoch-10 checkpoint and `RECOVERED_EPOCH10_GATE_COMPLETE`, then stops in `AWAITING_REVIEW`; it cannot access validation, enter epoch 11, or write `TRAINING_COMPLETE`.

This package implements a reviewable, fail-closed recovery path from the one admissible scientific input: the complete atomic epoch-9 checkpoint with SHA-256 `9aa3c1c1ad87889c730ff2ac0c936ed5b64ea23fd90c14ba3dbd16743046b2d4`. The immutable base configuration does not itself qualify a yaw floor, set a threshold, create a recovery experiment, or authorize training; the prospective amendment and separately bound authorization govern the single epoch-10 gate. Original epochs 10–26 are labeled `CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE` and every runtime resolves epoch 9 explicitly rather than searching for a latest checkpoint.

The only scientific numerical change is the shared training/inference yaw map:

`raw_fp32 / clamp_min(stable_l2_norm_fp32(raw_fp32), qualified_tau)`

The immutable base configuration retains the frozen candidate set `[1e-5, 1e-4, 1e-3, 1e-2]` and a null `selected_tau`; the prospective amendment alone fixes `tau=1e-2` for the epoch-10 gate. Ordinary nonzero rows at or above tau use the unchanged unit-normalization equation. Zero and near-zero rows have no invented direction or fallback. Diagnostics record raw norms and affected count/fraction. Dimension inference keeps the registered FP64 exponential with a representable-range/finiteness check and no physical clamp.

The post-accumulation/pre-step breaker computes gradient norms by the three registered optimizer groups plus global, momentum norms, exact proposed SGD update norms including weight decay/momentum, and maximum parameter-relative proposed update. It uses one-tensor-at-a-time scalar reductions and retains no FP64 copies or full proposed-update lists. The immutable base ceilings remain null; the prospective gate derives its ceilings from the verified replay union. It never clips, skips, or treats loss magnitude as a breaker criterion. Every required gradient must remain finite; isolated exact-zero gradients are logged, while the prospective gate requires each required trainable group to be observed nonzero at least once across the scientific epoch.

All executable paths are gated:

- replay requires explicit reviewed update-447 sample identities and an opt-in token;
- retrospective qualification is retired by the prospective protocol amendment;
- the prospective epoch-10 gate requires fixed replay evidence and authorization bound to the exact commit and package hash;
- scientific continuation additionally requires qualified artifacts and explicit user authorization bound to the same commit;
- a new continuation is create-only from epoch 9; explicit interruption recovery accepts only the latest contiguous, fully verified epoch-boundary checkpoint in that same output and refuses partial or overwrite-prone state;
- validation inference requires recovered epoch-26 completion;
- evaluation uses the frozen original evaluator for original epochs 3/8 and recovered epochs 16/22/26, with `v025_selected_only` sensitivity.

No command in the “Commands reserved for later” section of the review packet was executed during implementation.

See `IMPLEMENTATION_REVIEW_PACKET.md`, `IMPLEMENTATION_REVIEW_PACKET.json`, and `NUMERICAL_OPERATION_AUDIT.md`.
