# Radar-pps study — accuracy vs cost (2026-07-02)

| pps | veh IoU | mIoU | veh F1 | person F1 | person near-recall | payload KB(comp) | front_ms | back_ms | RTT_ms |
|---|---|---|---|---|---|---|---|---|---|
| 100k | 0.910 | 0.827 | 0.875 | 0.718 | 0.86 | 1073.6 | 49.7 | 8.9 | 42.0 |
| 150k | 0.943 | 0.837 | 0.850 | 0.742 | 0.75 | 1041.4 | 49.3 | 7.2 | 39.4 |
| 200k | 0.934 | 0.837 | 0.870 | 0.806 | 0.91 | 1048.3 | 49.7 | 8.5 | 41.1 |
| 250k | 0.925 | 0.835 | 0.851 | 0.783 | 0.85 | 1032.4 | 48.9 | 7.2 | 39.4 |
| 300k | 0.939 | 0.849 | 0.856 | 0.790 | 0.89 | 1059.8 | 49.8 | 7.3 | 40.3 |

Accuracy from the ablation (150k–300k same seed/route; 100k is the prior-collection reference).
Cost from loopback split-inference deployment (400 frames = 2 crowded loops, seed 31, same route).
back_ms / RTT are over the frames whose result returned; front_ms & payload are per-frame.

## Conclusion

**Higher radar pps buys pedestrian recall up to ~200k — at no transport cost.**

- **Accuracy:** pedestrian detection is the only radar-limited class. Person F1 climbs 0.72→0.81 and
  near-field recall 0.74 (150k) → 0.90 (200k), then plateaus. Vehicles and segmentation are already
  saturated (flat across pps). So ~200k pps is the accuracy sweet spot; beyond it adds nothing.
- **Transport cost is pps-independent.** Radar is fused as a fixed-size 4-channel raster *before* the
  split point, so the intermediate-tensor shape doesn't depend on point count: uncompressed payload is
  identical (2835 KB) for all five models, compressed varies only ~4% (content entropy, not size), and
  front/back/RTT latency is flat (~49 / ~8 / ~40 ms). There is **no accuracy–bandwidth tradeoff** on the
  wire — the cost of higher pps is entirely front-end (rasterizing more points), not the split link.
- **Bottom line:** run at ~200k pps — you get the full pedestrian-recall gain, pay no extra payload or
  latency for it, and gain nothing by pushing higher.

**Caveats:** loopback (not the live 5G link) — payload + front/back compute are exact, but the ~40 ms
RTT and transport estimate are localhost, not over-the-air; real 5G RTT is a follow-on with the OAI
stack up (`--role back` receiver). Also, ~14–20% of frames returned a result over UDP (the ~1 MB payload
fragments), so back/RTT means use 56–78 samples/model — enough for a stable mean, and the high UDP drop
rate for MB-scale tensors is itself a deployment note (large intermediate tensors want a reliable transport).

**Figures (PDF + PNG):** `cooperative_fusion/pps_study_figs/` — person_recall_by_distance,
latency_breakdown, cost_vs_pps, pareto_accuracy_vs_payload.
