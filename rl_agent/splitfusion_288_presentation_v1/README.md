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
  --output experiments/splitfusion_rl_policy_design_v1/20260910_288_results_presentation_pack_v6
```

After the final edge calibration, build the simulator-aware pack with:

```bash
python3 -m rl_agent.splitfusion_288_presentation_v1.build_final_simulated_pack
```

The final builder gives every network profile a separate validation sheet
(Figures 01a–01d, three panels each) and localization sheet (Figures 02a–02d,
four panels each), while keeping their frozen action-level values unchanged.
Marker size encodes profile-specific simulated installation probability.
Figures 04, 06, 07 and 08 use the counterfactual final-edge result; Figures 03,
05 and 09 retain the applicable measured evidence. Figure 10 uses the same
eight stages and colors as Figure 09 and pairs the original and final paths.
The self-contained architecture document embeds a portable SVG followed by
its Mermaid source, matching the LR-ASPP/FCOS report convention.
