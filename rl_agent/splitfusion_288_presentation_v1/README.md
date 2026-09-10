# SplitFusion 288-results presentation pack

This package builds a presentation-only sequence from the hash-bound 288-cell
consolidation, the original four-action timing diagnostic and the qualified
edge-optimization result. It deliberately omits the Phase-13C localhost plot.

The plots retain validation, campaign and follow-up timing evidence as distinct
domains. In particular, the 288-cell capture-to-install AoI is not relabelled
as pure network transport, UDP datagram accounting is not described as
retransmission, and stage medians are not presented as an additive end-to-end
latency statistic.

Run from the repository root:

```bash
python3 -m rl_agent.splitfusion_288_presentation_v1.build_pack \
  --output experiments/splitfusion_rl_policy_design_v1/20260910_288_results_presentation_pack_v4
```
