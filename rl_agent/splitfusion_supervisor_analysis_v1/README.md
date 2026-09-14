# SplitFusion supervisor analysis v1

This package turns the immutable 288-cell campaign and the qualified direct
edge-to-map counterfactual into presentation-ready, auditable artifacts.

It deliberately separates the recorded model front from four causal intervals:

1. **Model front backbone:** the model's recorded `front_backbone` duration.
2. **UE action path:** action start through tensor preparation, model front,
   compression and serialization.
3. **Feature uplink:** UE transmit-ready through complete edge reassembly.
4. **Edge-to-map service:** edge reassembly through direct spatial-map install.
5. **Action-to-map total:** action start through direct spatial-map install.

The first two clocks are joined only through the previously validated
same-host monotonic-to-wall clock bridge. Stage percentiles always carry their
own denominator; missing stages are never assigned a latency of zero.

The quality axis is a provisional presentation score, not the PPO reward:

$$
Q_{\mathrm{joint}}=
\sqrt{Q_{\mathrm{seg}}\sqrt{Q_{\mathrm{vehicle\ IoU}}
Q_{\mathrm{person\ box\ IoU}}}}.
$$

The geometric mean prevents strong segmentation from hiding poor localization
or the reverse. All constituent validation metrics remain in the CSV.

Run offline:

```bash
python3 -m rl_agent.splitfusion_supervisor_analysis_v1.build_analysis
```

No CARLA, OAI, Docker, RFsim, CUDA, or model inference is launched.

`profiled_sensor_stages.py` supplies bit-preserving P1–Pn equivalents for a
later short live diagnostic. It separates radar coordinate conversion,
stationary tracking, camera projection, rasterization and evidence packaging;
it also separates camera conversion/resize/packing, host-to-device copies,
normalization, radar resize/packing and seven-channel concatenation. The CUDA
version synchronizes timing events and is therefore diagnostic-only. It is not
installed into the hash-pinned deployment path by this offline analysis.
