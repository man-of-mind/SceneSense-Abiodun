# Task A — argmax-stability / rank-reversal result

**Verdict:** `NO_PRACTICAL_REVERSAL_ON_AVAILABLE_CONTEXTS`.

The exact registered intersection contains **1,683 sample IDs** and all 36 published profiles. Per-frame segmentation was already present and validated structurally, so incremental segmentation re-evaluation cost was **0 GPU-minutes**. A clean 36-profile regeneration is estimated at 35–45 GPU-minutes from the recorded 72-profile runtime.

The primary result is seg-inclusive reward-v5 utility. Detection-only results are retained as a diagnostic, not allowed to close Phase 1 on their own.

## Strongest primary-family cells

| context_family | budget_kib | global_profile | action_change_fraction | mean_utility_lift | ci95_low | ci95_high | p_holm | practical_reversal |
|---|---|---|---|---|---|---|---|---|
| class_mix | 64.1 | ae64__uint4__roi0.5 | 0.3911 | 0.00817 | 0.00154 | 0.01495 | 0.22316 | False |
| nearest_range | 64.1 | ae64__uint4__roi0.5 | 0.4284 | 0.00813 | 0.00302 | 0.01334 | 0.05039 | False |
| nearest_range | 341.0 | ae128__uint6__roi0.0 | 0.6799 | 0.00289 | 0.00023 | 0.0059 | 0.4773 | False |
| nearest_range | 289.0 | ae128__uint6__roi0.0 | 0.6799 | 0.00289 | 0.00031 | 0.00598 | 0.46711 | False |
| nearest_range | 1050.3 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00025 | 0.00591 | 0.4773 | False |
| nearest_range | 784.8 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00019 | 0.00572 | 0.5075 | False |
| nearest_range | 759.7 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00036 | 0.00585 | 0.39192 | False |
| nearest_range | 598.1 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00033 | 0.00588 | 0.42232 | False |
| nearest_range | 565.6 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00032 | 0.00595 | 0.39192 | False |
| nearest_range | 452.2 | ae128__uint6__roi0.0 | 0.6799 | 0.00288 | 0.00021 | 0.00585 | 0.4773 | False |

## Interpretation

No available primary context cleared the pre-registered held-out practical gate. Because segmentation was included, this is stronger than a detection-only null, but remains scoped to the available class/range contexts and measured profiles. True occlusion, cyclists, and broader scenarios were not tested.

`edge_truncated_present` is an image-boundary truncation proxy, not an occlusion label. `reference_low_confidence_or_miss` uses the full-quality reference output and is supporting/non-deployable.

## Artifacts

See `budget_context_results.csv`, `context_winners.csv`, `frame_context.csv`, `profile_costs.csv`, `resolved_config.yaml`, `manifest.json`, and `figures/`.
