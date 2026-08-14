# Task B — observed vulnerable-object guardrail ablation

The hard rules prevent SKIP whenever an observed pedestrian/cyclist is active and clamp low-confidence vulnerable-object frames to ROI0. They cannot protect detector misses or unrepresented hidden hazards.

C1 remains dominant: if no C1-admitted action satisfies the vulnerable rule, the least-risk C1 action is used and `vulnerable_guardrail_unachievable` is raised.

## Primary paired cost (confidence < 0.30)

- Action changes: 26.04% of held-out ticks.
- Finite matched-reward delta (on - off): -0.047677, trajectory-cluster 95% CI [-0.071720, -0.007371].
- Offered-load delta: +1.0994 Mbps; selected-payload delta: +21.904 KiB.
- Matched-safe-rate delta: +1.630 percentage points.

## Threshold sensitivity

```text
        guardrail_variant  frames  vulnerable_opportunity_frames  low_confidence_opportunity_frames  mean_matched_reward_finite  mean_offered_mbps  mean_payload_kib_selected  split_pct  shield_feasible_pct  matched_safe_pct  skip_on_observed_vulnerable_count  roi_drop_on_low_confidence_vulnerable_count  guardrail_applied_frames  guardrail_unachievable_frames
                 disabled    2638                            567                                312                    0.196545           0.155195                   3.058984   3.980288            72.517058         91.129644                                535                                           17                         0                              0
        enabled_conf_0.20    2638                            567                                110                    0.147695           1.193063                  23.685595  29.378317            85.898408         92.797574                                  0                                            0                       567                              0
enabled_conf_0.30_primary    2638                            567                                312                    0.148868           1.254618                  24.963002  29.378317            85.822593         92.759666                                  0                                            0                       567                              0
        enabled_conf_0.40    2638                            567                                364                    0.150107           1.256935                  25.547839  29.378317            85.708870         92.835481                                  0                                            0                       567                              0
```

Cyclists are supported by the rule but absent from the accepted corpus, so cyclist protection is contract-tested rather than empirically costed here.
