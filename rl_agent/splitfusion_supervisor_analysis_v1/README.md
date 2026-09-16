# SplitFusion supervisor analysis v1

This package turns the immutable 288-cell campaign and the subsequently
live-validated sensor, edge, latest-only scheduling, and direct edge-to-map
optimizations into presentation-ready, auditable counterfactual artifacts.

It deliberately separates these causal intervals:

1. **Optimized sensor compute:** production sensor-preparation work before
   seven-channel concatenation, excluding CARLA sample wait.
2. **UE action path:** seven-channel concatenation through model front,
   ranker/selection, compression, serialization, and UDP send completion.
3. **Feature uplink:** send completion through complete edge reassembly.
4. **Early edge control boundary:** reassembly through latest-only scheduler
   wait, feature reconstruction and synchronized model-tail completion.
5. **Early controller-ready total:** production sensor-compute start through
   model-tail completion at the edge.
6. **Physical map endpoints retained for audit:** reassembly/action/capture
   through authoritative map install.

The CSV retains `edge_map_*`, `total_*` and `capture_total_*` for physical
freshness accounting. Figures 03 and 04 instead study the proposed early
model-tail-complete boundary. That boundary is feedback-ready at the edge: it
does not include the unmeasured compact-ACK return trip and does not prove that
post-processing or map installation succeeded.

The first two clocks are joined only through the previously validated
same-host monotonic-to-wall clock bridge. Stage percentiles always carry their
own denominator; missing stages are never assigned a latency of zero.

Cross-profile latency bars use one common action support across every displayed
stage so adverse conditions cannot appear faster merely because heavy actions
have no conditional latency sample. They require at least 100 frame samples per
stage/profile so their empirical P99 is not a one-frame maximum. Figures 02--04
plot P50 points only with at least 10 frame samples and label withheld and
missing action counts explicitly; every lower-support value remains in the
CSV. Figure 07 reports complete application
reassembly, model-tail completion and map installation per frame sent over all
72 actions; this keeps the latency clouds from hiding missing outcomes.
Figure 06 numbers its displayed production functions sequentially from P01 to
P18; `sensor_optimization_stage_percentiles.csv` retains the live profiler's
original identifier in `source_stage`.

The presentation figures keep the quality dimensions separate:
aggregate semantic-segmentation mIoU, vehicle overlap IoU, person box-mask
IoU, vehicle centroid XY error, and person centroid XY error. A sixth figure
in each group restores the joint model-quality coordinate below. The retained
evidence does not contain separate vehicle/person semantic-segmentation mIoU;
the two object-overlap IoUs must not be relabelled as class-specific semantic
segmentation.

The CSV also retains a provisional combined presentation score, not the PPO
reward:

$$
Q_{\mathrm{overlap}}=\sqrt{\mathrm{IoU}_{\mathrm{vehicle}}
\mathrm{IoU}_{\mathrm{person}}},\qquad
Q_{xy}=\exp\left(-\frac{\sqrt{(e_v^2+e_p^2)/2}}{1\,\mathrm m}\right),
$$

$$
Q_{\mathrm{loc}}=\sqrt{Q_{\mathrm{overlap}}Q_{xy}},\qquad
Q_{\mathrm{joint}}=\sqrt{mIoU_{\mathrm{seg}}Q_{\mathrm{loc}}}.
$$

The geometric mean prevents strong segmentation from hiding poor localization
or the reverse. All constituent validation metrics remain in the CSV.

Run offline:

```bash
python3 -m rl_agent.splitfusion_supervisor_analysis_v1.build_analysis
```

No CARLA, OAI, Docker, RFsim, CUDA, or model inference is launched.

`profiled_sensor_stages.py` supplied bit-preserving P1–Pn equivalents for the
completed live sensor diagnostic. It separates radar coordinate conversion,
stationary tracking, camera projection, rasterization and evidence packaging;
it also separates camera conversion/resize/packing, host-to-device copies,
normalization, radar resize/packing and seven-channel concatenation. The CUDA
version synchronizes timing events and is therefore diagnostic-only. It is not
installed into the hash-pinned deployment path by this offline analysis. The
validated optimized distribution is applied by equal-percentile mapping, not
by subtracting a constant from every historical frame. Measured 288-cell radio
reassembly/admission outcomes remain fixed.
