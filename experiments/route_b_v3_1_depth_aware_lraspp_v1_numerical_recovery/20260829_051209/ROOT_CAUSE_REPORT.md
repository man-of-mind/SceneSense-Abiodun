# Route B v3.1 depth-aware LR-ASPP numerical root-cause report

## Deterministic-reproduction gate

Both pristine full-FP32 CUDA processes used the failed implementation at `049f7029d9156871e02b2aed34da3cdbcbd842ef`, scientific seed `20260829`, sampler seed `20260830`, physical batch 16, accumulation 1, the original AdamW/LR schedule, the official weight, and the hash-validated read-only train cache.

Initial model, parameter, buffer, optimizer and RNG hashes were identical. All ordered sample IDs, inputs and targets for batches 1–15 were byte-hash identical. Batch 14 matched the failed record exactly:

- full batch SHA-256: `682263c0265112e211e47ea92698e3860c6b19af7095ece0032af3c6879b3f35`
- input SHA-256: `0f9cfc19cb7fe9ec6619fe0f563ca2a1f9b9afdf7aad9fe1d8f0dac39433006d`
- targets SHA-256: `30521d548c1b991d67b57eab0d2d2dbcbf3f53def2ec90cb73e368089abd51a1`
- ordered sample-ID SHA-256: `d741aee44fb23303069f393e4dbc407af3ae9c39eb917da72a09b266e348cae9`

The clean replays nevertheless diverged during update 1. Their combined segmentation losses differed by one FP32 quantum (`14.79089641571045` versus `14.790901184082031`, delta `-4.76837158203125e-06`); pre-clip and post-clip gradient hashes and the resulting AdamW hashes differed. CUDA emitted explicit warnings that the registered cross-entropy forward and grid-sampler backward paths do not have deterministic implementations. The reported clip norm happened to agree (`3630824.5`), but tensor hashes did not.

After update 13, buffers and RNG remained identical, while persistent scientific state did not:

| State | Reproduction 1 | Reproduction 2 |
|---|---|---|
| Model | `294df546cf44010c116bfdcebec1fa528c23752d1eadf750f5ee1703a20f57d7` | `c9314d3e6190747cc994d19ba8690277d012aeacce9eaf2a86eb3da256c05ca9` |
| Parameters | `7856579271faf25b61490bef1f5941cbd171972b029ac11a66fd6b18e82d0a99` | `17370a10f07fa27d26744eb4915261ebc3013bfb5e2eb468658f34d3e01e4ed1` |
| AdamW | `3cb48b62ae8df123b7ec45f41b11e0e012329b74bcaabb0b214b5a8ea3971f7e` | `6aa8973fb9675dd8a49a27dae852184f55006f0f6afb1493310117931ece5962` |
| Buffers | `36bab059ade3ad769a549c33d6fc24724b09d60c79e90195592b65c417d8cb77` | same |
| RNG | `4f20a92fc72170ff89f38a3dd4bfbc28956d0b8ce15358d9d260bad3103812f4` | same |

The protocol requires mutually identical post-update-13 hashes and tensor summaries. That gate fails even though both paths reach the same categorical overflow.

## First non-finite operation

In both replays the first non-finite operation was sequence 240, `vehicle.dimensions.exp`. Every model/module output and every preceding functional operation was finite. In particular, vehicle depth `expm1` was finite for all 44 owners (maximum 36.8705/36.9269 m), and person depth `expm1` was finite for all 15 owners (maximum 116.5981/116.6502 m). The registered unbounded depth formulation did not cause this failure.

The overflowing prediction belonged to:

- sample: `canonical_v3_02_train_50_50_s502_tm1502_001305_frame5438`
- object: `canonical_v3_02_train_50_50_s502_tm1502:actor:82`
- head: vehicle `log_dimensions`, component 0
- native cell: `(y=53, x=87)`
- target forward depth: `14.780355003234902 m`
- target dimensions: `[5.006043434143066, 1.8809516429901123, 1.540313720703125] m`

Reproduction 1 supplied finite log-dimension values `[95.49971771240234, -43.13870620727539, 31.307249069213867]`; reproduction 2 supplied `[95.54299926757812, -43.089176177978516, 31.277645111083984]`. FP32 `exp` overflowed component 0 in each run. One of 132 vehicle dimension values became positive infinity; the subsequent clamp and log preserved it, the macro dimension loss became non-finite, and then the total loss became non-finite.

The implemented dimension loss evaluates `log_dimensions -> exp -> clamp_min -> log -> SmoothL1`. On its safe finite domain, direct `SmoothL1(predicted_log_dimensions, log(target_dimensions))` is the obvious algebraically equivalent Class-A candidate. It was not registered, applied or qualified because the prerequisite deterministic-reproduction gate failed. No depth bound, residual bound, overflow bin, initialization change, loss/optimizer/schedule change, batch change, or sample exclusion was attempted.

## Outcome

The exact first overflow category and originating object reproduced twice, but the two clean scientific state trajectories and causal tensor values were not mutually identical. Under the frozen stopping rule, no repair is authorized and the terminal outcome is `DEPTH_AWARE_NONDETERMINISTIC_FAILURE`.
