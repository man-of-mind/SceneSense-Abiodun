# Route B v3.1 depth-aware LR-ASPP numerical recovery — final report

**Terminal verdict: `DEPTH_AWARE_NONDETERMINISTIC_FAILURE`**

The original batch-14 non-finite category reproduced in both pristine full-FP32 CUDA replays, but the two scientific state trajectories were not mutually deterministic. The protocol requires identical initial, batch, post-update-13, and operation-summary evidence before authorizing a repair. Initial state and all 15 ordered batches matched; gradients diverged at update 1 and post-update-13 model/AdamW hashes did not. No repair was registered or applied, and scientific attempt 2 was not started.

## Immutable evidence

- Failed implementation: `049f7029d9156871e02b2aed34da3cdbcbd842ef` on local `master`.
- Failed experiment remained unchanged. Its runtime-failure SHA-256 is `7e78d77c19ef4d09e1af42387526f3bd5b87b23e8f2808303d047532243b826b`; final-report SHA-256 is `d30e52853435b5a901662eb53e8efd3d8b01f4d0b610a3376e2cf25bcd333b4c`.
- Official pretrained SHA-256: `5c1a416349c4cf298f2a6a5e2600ed0ee55e604713578f5e74e6bc8bcaef7997`.
- Manifest SHA-256: `5d65e6eb14aadea11ca6bab6e82f0c94c31a50746611d167d282d8988a4504c2`.
- Every read-only cache shard was rehashed successfully; the cache was reused without rebuilding or mutation.
- Reproduction artifact hashes: run 1 `043adfe55aa3638b5e399eed6ffc44eb65558988157728c45fea1104ba5bc467`; run 2 `8c65085efde75f329ca613710afbd7d96a200a99ab1a7ebd58cd60a6bfca1927`.

## Failed-batch identity and deterministic comparison

The two replays matched the recorded 16 batch-14 indices and sample IDs. Batches 1–15 had identical ordered IDs, inputs, and targets. Batch 14 hashes were:

| Component | SHA-256 |
|---|---|
| Full batch | `682263c0265112e211e47ea92698e3860c6b19af7095ece0032af3c6879b3f35` |
| Seven-channel input | `0f9cfc19cb7fe9ec6619fe0f563ca2a1f9b9afdf7aad9fe1d8f0dac39433006d` |
| Targets | `30521d548c1b991d67b57eab0d2d2dbcbf3f53def2ec90cb73e368089abd51a1` |
| Ordered IDs | `d741aee44fb23303069f393e4dbc407af3ae9c39eb917da72a09b266e348cae9` |

Initial model state (`425e33aff81cb26e99b980ff12dfe1ba42748b8aa6dd492cc4dbf1d027f2b95c`), optimizer, parameters, buffers, and RNG hashes were identical. At update 1, combined segmentation loss differed by `4.76837158203125e-06`; pre/post-clip gradient and AdamW hashes differed. CUDA explicitly warned that the registered cross-entropy forward and grid-sampler backward paths lack deterministic implementations.

| Post-update-13 state | Replay 1 | Replay 2 | Equal |
|---|---|---|:---:|
| Model | `294df546cf44010c116bfdcebec1fa528c23752d1eadf750f5ee1703a20f57d7` | `c9314d3e6190747cc994d19ba8690277d012aeacce9eaf2a86eb3da256c05ca9` | no |
| AdamW | `3cb48b62ae8df123b7ec45f41b11e0e012329b74bcaabb0b214b5a8ea3971f7e` | `6aa8973fb9675dd8a49a27dae852184f55006f0f6afb1493310117931ece5962` | no |
| Buffers | `36bab059ade3ad769a549c33d6fc24724b09d60c79e90195592b65c417d8cb77` | same | yes |
| RNG | `4f20a92fc72170ff89f38a3dd4bfbc28956d0b8ce15358d9d260bad3103812f4` | same | yes |

## First non-finite operation

Both replays identified sequence 240, vehicle `exp(log_dimensions)`, as the first non-finite operation. It originated from object `canonical_v3_02_train_50_50_s502_tm1502:actor:82` in sample `canonical_v3_02_train_50_50_s502_tm1502_001305_frame5438`, native cell `(y=53,x=87)`, component 0. The target dimensions were `[5.006043434143066, 1.8809516429901123, 1.540313720703125] m` at forward depth `14.780355003234902 m`.

Replay 1’s finite raw log dimensions were `[95.49971771240234, -43.13870620727539, 31.307249069213867]`; replay 2’s were `[95.54299926757812, -43.089176177978516, 31.277645111083984]`. FP32 exponentiation overflowed component 0. The following clamp and log preserved infinity, making the dimension loss and total loss non-finite. Every earlier probed operation was finite. Vehicle and person depth `expm1` were finite, so the registered unbounded 32-bin depth contract was not the cause.

## Repair and qualification status

- Single repair: **none authorized**.
- Repair classification: **none**. Direct log-space Smooth-L1 is an apparent Class-A candidate for the redundant `exp -> clamp -> log` dimension-loss round trip, but the deterministic prerequisite failed, so it was not registered, changed, tested, or qualified.
- Semantic-equivalence test: not run.
- Repaired batch-14/15 qualification: not run.
- Disposable full epoch-1 qualification: not run.
- Repair-source commit: **none**.
- Numerical-audit commit: `b6bcc08341d8e897fd3d9e5a854cb60b419ecb77`.

## Scientific and evaluation accounting

| Item | Status |
|---|---|
| Scientific attempt 1 | Failed at epoch 1 batch 14 under commit `049f702`; preserved unchanged |
| Diagnostic pristine replays | Two; both non-candidate, both discarded |
| Scientific attempt 2 | Not started; repair gate failed |
| `INITIAL_STATE.pt` for attempt 2 | None |
| Epoch 10 checkpoint/evaluation | None |
| Epoch 20 checkpoint/evaluation | None |
| Epoch 30 checkpoint/evaluation | None |
| Epoch 40 checkpoint/evaluation | None |
| Original selection outcome | None; no repaired scientific checkpoints |
| v0.25 sensitivity | Not licensed or run |

No validation data, validation depth, test data, CARLA, OAI contents/execution, q/AE artifacts, hybrid-q experiments, live runtime, or 288 measurements were accessed. No branch, push, remote access, architecture variant, hyperparameter change, depth/residual bound, overflow class, batch change, or scientific retry occurred. The pre-existing dirty `OAI/openairinterface5g` submodule remains the only dirty repository entry outside ignored recovery artifacts.

Desktop notification was delivered successfully. The terminal verdict and completion sentinel agree. This report is included in a dedicated final-report commit; its hash is reported in the external handoff to avoid self-reference.
