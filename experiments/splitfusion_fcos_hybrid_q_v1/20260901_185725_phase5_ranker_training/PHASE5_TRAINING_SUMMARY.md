# Hybrid-q Phase 5 — ranker training and train-holdout checkpoint selection

**Terminal: `ROI_DROP_NOT_SAFE_ON_TRAIN_HOLDOUT`.** One 12-epoch scientific training run,
one q=0 holdout baseline and nine checkpoint/q holdout evaluations. No checkpoint/q pair
passed every registered preservation gate, so no checkpoint was selected and validation was
not opened.

| | |
| --- | --- |
| artifact | `experiments/splitfusion_fcos_hybrid_q_v1/20260901_185725_phase5_ranker_training` |
| training | 81.2 min, 10,164 optimizer updates (847 x 12), 6,776 q-aware |
| holdout evaluation | 10 passes over 3,284 reserved frames |
| device | RTX 5090; training peak allocated 25,140 MiB, evaluation peak 2,976 MiB |
| seed / augmentation | 20260829 / disabled |

## 1. Bound inputs

Every input was verified by exact hash before any weight was touched. The teacher-cache
manifest hash binds all 66 shard hashes, so verifying the manifest and then each shard
against it is one closed identity chain.

| input | sha256 |
| --- | --- |
| perception forward lock (p025) | `86d6f13a…d943fe1` |
| frozen epoch-26 checkpoint | `da14d21e…a160a297f` |
| hybrid-q locked configuration | `b2b0d842…b150ebfcde6` |
| Phase-4 teacher-cache manifest | `e1ef600e…acd7fc273` |
| Phase-4 fit-reference medians | `f91570c7…bcf552729e` |
| teacher-cache shards | 66 / 66 verified, 1,452,208,370 B |

Frame identity: **13,543 fit / 3,284 holdout**, ordered-sample-id digests equal to the locked
`3e20ccee…fa252e` and `8c7c4cc6…754f0a`. The cached set equals the registered train split
exactly, with **0** intersection against validation; the manifest contains no test split at
all. No duplicate and no missing frame IDs. The four frozen medians match both
`fit_reference_medians.json` and the contract constants exactly: D `0.7242346405982971`,
G `0.10442040860652924`, S `0.09630495309829712`, A `0.013652884401381016`.

All 1.35 GiB of teacher maps (16,827) were preloaded once into host memory. No C2 tensor was
cached. Holdout maps are loaded so the cache is verified end to end, but the accessor refuses
to return one, so no holdout supervision can reach an optimizer batch.

Every Phase-1..4 module that defines the cached teacher semantics — `ranker.py`,
`selection.py`, `codec.py`, `guards.py`, `training.py`, `teacher_cache.py`,
`gpu_qualification.py`, `locked_config.json`, `__init__.py` — is **bit-identical** to what the
cache recorded. Only `contract.py` (Phase-5 constants appended), `README.md` and the three new
Phase-5 runners differ, and the runner fails closed if any frozen-semantics module changes.

## 2. Training execution and integrity

| gate | result |
| --- | --- |
| 12 epochs x 847 updates x 13,543 frames | 10,164 updates, every fit frame exactly once per epoch |
| deterministic epoch shuffle, seed 20260829 + epoch | seeds 20260830..20260841, 12 distinct order digests |
| physical/effective batch 16, no accumulation, final partial batch retained | 846 x 16 + 1 x 7 |
| holdout frames in an optimizer batch | **0** |
| gradient qualification (all five named ranker tensors) | qualified every epoch; 0 zero-gradient, 0 missing-gradient batches |
| optimizer ownership | exactly 2,144 ranker parameters, ranker only |
| frozen perception unchanged at epochs 4, 8, 12 | 371 / 371 parameters and buffers bit-identical |
| frozen perception gradients | none, on every update |
| q cycle 0.30 -> 0.50 -> 0.70, continuous across epochs | 282/283 per q per epoch, never reset |
| G excluded when a batch has zero geometry supervision | rule active; **0** such batches occurred |

Zero G-exclusions is the correct consequence of shuffling, not a dormant rule. Phase 4's 13
zero-geometry batches arose in *contiguous* registered order; under a random shuffle a batch
needs all 16 frames to lack a matched FCOS location, which has probability ~0.043^16.

A bounded pre-flight smoke exercised both stages on the real data, mask, loss and optimizer
path using a ranker that was then discarded. Nothing it produced entered the scientific run.

## 3. The q-aware stage diverged

| ep | stage | total | distill | clipped | max pre-clip \|\|g\|\| | D | G | S | A |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | distillation | 9.3206 | 9.3206 | 0.0% | 0.93 | – | – | – | – |
| 4 | distillation | 9.1613 | 9.1613 | 0.0% | 0.81 | – | – | – | – |
| 5 | q-aware | 31.13 | 250.21 | 98.0% | 224.6 | 1.234 | **10.330** | 3.181 | 9.706 |
| 8 | q-aware | 134.91 | 1222.46 | 100% | 653.1 | 1.889 | **34.442** | 5.474 | 8.838 |
| 12 | q-aware | 245.31 | 2288.49 | 100% | 1116.0 | 2.216 | **47.792** | 6.283 | 9.558 |

Ranker weight norm at the three candidates: **5.87 -> 31.43 -> 40.56**; the output
`score.weight` alone goes 0.96 -> 6.42 -> 9.42.

Mechanism. The hard exact-cardinality mask is scale-invariant, so the masked task term is
indifferent to score magnitude; the only thing resisting magnitude growth is
`0.1 x` listwise cross-entropy, and it loses. Constant LR `1e-3` with essentially every update
clipped at norm 5.0 produces near-constant-magnitude steps, so the parameters grow roughly
linearly and the softmax cross-entropy inflates without the retained cell set improving.

The damaging consequence is that **the q-aware stage made masked geometry monotonically worse
and ended worse than a randomly initialized ranker.** The pre-flight smoke measured a fresh
ranker at normalized G ~ 39.4 (3 batches, indicative only); the distillation-trained ranker
entering stage B sat at 10.330; eight q-aware epochs drove it back up to 47.792. G dominates
the objective throughout.

This is a property of the registered configuration, not a numerical fault. Stage A is
well-behaved throughout (loss decreasing, 0% clipping, gradient norm < 1), an untrained
ranker's distillation loss is exactly `log 21504 = 9.978` as it must be, and every runtime and
ownership gate held on all 10,164 updates. LR, clip and the 0.1 weight are locked and were
not retuned.

Stage A itself learned little: 9.3206 -> 9.1613 against a uniform reference of 9.978, i.e.
about 0.8 nats better than uniform after four epochs.

## 4. q=0 holdout baseline reproduces the frozen p025 result exactly

Computed once over the 3,284 reserved frames through the exact p025 service pipeline.

| field | published p025 | Phase-5 q=0 | |
| --- | --- | --- | --- |
| observable GT (AVO>=0.65) | 2556 | 2556 | exact |
| TP / FP / FN | 2249 / 253 / 307 | 2249 / 253 / 307 | exact |
| AVO-ignored / structurally ignored GT | 2147 / 21054 | 2147 / 21054 | exact |
| person precision | 0.898880895283773 | 0.898880895283773 | exact |
| person recall | 0.8798904538341158 | 0.8798904538341158 | exact |
| person F1 | 0.8892843020956901 | 0.8892843020956901 | exact |
| person XY MAE m | 0.5346737056220642 | 0.5346798001323065 | 6.1e-06 |

The counts and all three rates are bit-identical; XY MAE differs by 6.1e-06 m because the
published run reconstructed candidates from the consolidation cache while Phase 5 runs a live
model forward. This is an independent confirmation that the Phase-5 evaluation is the frozen
p025 pipeline and not a re-implementation of it. Separately, the person scorer used here was
required to agree with the frozen `qualification.score_view` on all 13 shared fields, and did.

Full q=0 baseline: vehicle P 0.966722 / R 0.974723 / F1 0.970706 / XY MAE 0.373728 m
(TP 8715, FP 300, FN 226 over 8,941 eligible); person AVO>=0.65 P 0.898881 / R 0.879890 /
F1 0.889284 / XY MAE 0.534680 m; vehicle IoU 0.920036, person box-mask IoU 0.668245,
foreground mIoU 0.794141; person 20-40 m recall 0.818571 (1340 / 1637).

## 5. Checkpoint/q holdout results

Retained cells and framed payload were exactly as registered on every pass: q=0 21,504 cells /
22,020,140 B / ratio 1.000000; q=0.30 15,053 / 15,417,004 / 0.700132; q=0.50 10,752 /
11,012,780 / 0.500123; q=0.70 6,451 / 6,608,556 / 0.300114. Framed encode/decode was verified
bit-exact on sampled frames of every pass.

| config | v P | v R | v F1 | v MAE | p P | p R | p F1 | p MAE | v IoU | p IoU | fg mIoU | p R 20-40 | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q=0 | .9667 | .9747 | .9707 | .3737 | .8989 | .8799 | .8893 | .5347 | .9200 | .6682 | .7941 | .8186 | – |
| ep4 q.30 | .9678 | .9719 | .9699 | .3876 | .8803 | .8920 | .8861 | .5424 | .8821 | .6524 | .7672 | .8369 | 4/12 |
| ep4 q.50 | .9672 | .9678 | .9675 | .4176 | .8551 | .8842 | .8694 | .5441 | .8412 | .6262 | .7337 | .8241 | 5/12 |
| ep4 q.70 | .9600 | .9498 | .9549 | .4664 | .8353 | .8772 | .8557 | .5851 | .7350 | .5948 | .6649 | .8155 | 9/12 |
| ep8 q.30 | .9463 | .8753 | .9094 | .5415 | .6587 | .7081 | .6825 | .7158 | .7481 | .4620 | .6050 | .6182 | 12/12 |
| ep8 q.50 | .9122 | .6394 | .7518 | .6914 | .5495 | .4844 | .5149 | .8292 | .4798 | .2749 | .3774 | .4197 | 12/12 |
| ep8 q.70 | .8571 | .2884 | .4316 | .9022 | .4765 | .2660 | .3415 | .8559 | .2113 | .1315 | .1714 | .2291 | 12/12 |
| ep12 q.30 | .9306 | .8291 | .8769 | .5946 | .5988 | .6796 | .6366 | .7117 | .7222 | .4354 | .5788 | .5925 | 12/12 |
| ep12 q.50 | .8783 | .4866 | .6263 | .7959 | .5200 | .4280 | .4695 | .8757 | .3909 | .2283 | .3096 | .3787 | 12/12 |
| ep12 q.70 | .8125 | .1512 | .2550 | 1.0219 | .4653 | .1991 | .2789 | .9892 | .1295 | .0930 | .1113 | .1790 | 12/12 |

`p` rows are person at AVO>=0.65. The epoch-8 and epoch-12 candidates fail every gate at every
q, consistent with the divergence in section 3: the q-aware stage produced rankers that are
worse than the distillation-only checkpoint it started from.

## 6. Decision

No checkpoint/q pair passed. Terminal `ROI_DROP_NOT_SAFE_ON_TRAIN_HOLDOUT`; no checkpoint was
selected; validation was not accessed. Selection did not consider teacher loss, training loss
or isolated metric improvements.

The closest pair is **epoch 4 at q=0.30** (worst absolute degradation 0.037936), and its
failure pattern is the substantive finding:

| gate | baseline | candidate | degradation | bound | |
| --- | ---: | ---: | ---: | ---: | --- |
| vehicle IoU | 0.920036 | 0.882099 | +0.037936 | 0.01 | **fail** |
| foreground mIoU | 0.794141 | 0.767237 | +0.026904 | 0.01 | **fail** |
| person AVO precision | 0.898881 | 0.880309 | +0.018572 | 0.015 | **fail** |
| person box-mask IoU | 0.668245 | 0.652374 | +0.015872 | 0.01 | **fail** |
| vehicle XY MAE m | 0.373728 | 0.387573 | +0.013845 | 0.05 | pass |
| person XY MAE m | 0.534680 | 0.542362 | +0.007682 | 0.05 | pass |
| person AVO F1 | 0.889284 | 0.886125 | +0.003159 | 0.015 | pass |
| vehicle recall | 0.974723 | 0.971927 | +0.002796 | 0.01 | pass |
| vehicle F1 | 0.970706 | 0.969866 | +0.000840 | 0.01 | pass |
| vehicle precision | 0.966722 | 0.967814 | **-0.001092** | 0.01 | pass |
| person AVO recall | 0.879890 | 0.892019 | **-0.012128** | 0.015 | pass |
| person AVO 20-40 m recall | 0.818571 | 0.836897 | **-0.018326** | 0.03 | pass |

**Every object-level detection and localization gate passes at 0.700132 of the framed q=0
payload; all four failures are segmentation or segmentation-coupled.** Vehicle precision,
person recall and person 20-40 m recall all *improve* slightly. The person-precision failure
is mechanistically downstream of segmentation rather than independent: the locked p025 person
consolidation gates candidates on a `semantic_support_threshold` of 0.10 over the semantic
logits, so degraded segmentation weakens that gate and admits extra person false positives.

This corroborates the existing repository finding in `rl_agent/density_knob/DENSITY_KNOB_RESULTS.md`
that ROI drop destroys segmentation. Phase 5 adds that on this frozen split the sparse,
object-centric heads tolerate a 30% spatial drop while the dense per-pixel semantic head does
not, because dropped cells decode to exact zeros that the semantic head cannot recover.

## 7. Scope

This artifact measures perception preservation only. Payload and zstd acceptance are deferred;
q=0.90 and q=0.98 were not evaluated; no quantization, AE, CARLA, OAI or continuous-q work was
run; the validation split was not opened and the locked test split does not exist in this
manifest.

What this does **not** establish. It does not show that ROI drop is unachievable. The q-aware
stage under the locked hyper-parameters diverged and ended worse than random initialization, so
the eight q-aware epochs are not a fair test of the hypothesis they were meant to test; the
only well-behaved candidate is the distillation-only epoch-4 checkpoint, which was never
trained with the mask in the loop. A bounded, pre-registered change to the q-aware stage — for
example a score-scale constraint, a lower or decayed LR, or a larger distillation weight — is
untested and therefore not falsified by this result. Any such change requires an explicit
decision and a new registration; it is not authorized here and no retune was attempted.

The measured segmentation sensitivity at epoch 4 / q=0.30 is the one result that does not
depend on the divergence, because that checkpoint predates the q-aware stage entirely.
