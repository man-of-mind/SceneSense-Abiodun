# SplitFusion FCOS V1 numerical recovery

State: `UNQUALIFIED_IMPLEMENTATION_ONLY`.

This package implements a reviewable, fail-closed recovery path from the one admissible scientific input: the complete atomic epoch-9 checkpoint with SHA-256 `9aa3c1c1ad87889c730ff2ac0c936ed5b64ea23fd90c14ba3dbd16743046b2d4`. It does not qualify a yaw floor, set a threshold, create a recovery experiment, or authorize training. Original epochs 10–26 are labeled `CORRUPTED_FINITE_GRADIENT_TRAJECTORY_DO_NOT_USE` and every runtime resolves epoch 9 explicitly rather than searching for a latest checkpoint.

The only scientific numerical change is the shared training/inference yaw map:

`raw_fp32 / clamp_min(stable_l2_norm_fp32(raw_fp32), qualified_tau)`

The candidate set is frozen at `[1e-5, 1e-4, 1e-3, 1e-2]`; `selected_tau` is deliberately null. Ordinary nonzero rows at or above tau use the unchanged unit-normalization equation. Zero and near-zero rows have no invented direction or fallback. Diagnostics record raw norms and affected count/fraction. Dimension inference keeps the registered FP64 exponential with a representable-range/finiteness check and no physical clamp.

The post-accumulation/pre-step breaker computes gradient norms by the three registered optimizer groups plus global, momentum norms, exact proposed SGD update norms including weight decay/momentum, and maximum parameter-relative proposed update. Its ceilings are deliberately null until qualification constructs each as ten times the healthy maximum. It never clips, skips, or treats loss magnitude as a breaker criterion. Zero gradients remain diagnostic unless an independently qualified reachability rule applies.

All executable paths are gated:

- replay requires explicit reviewed update-447 sample identities and an opt-in token;
- qualification requires independent review bound to the exact commit and package hash;
- scientific continuation additionally requires qualified artifacts and explicit user authorization bound to the same commit;
- validation inference requires recovered epoch-26 completion;
- evaluation uses the frozen original evaluator for original epochs 3/8 and recovered epochs 16/22/26, with `v025_selected_only` sensitivity.

No command in the “Commands reserved for later” section of the review packet was executed during implementation.

See `IMPLEMENTATION_REVIEW_PACKET.md`, `IMPLEMENTATION_REVIEW_PACKET.json`, and `NUMERICAL_OPERATION_AUDIT.md`.
