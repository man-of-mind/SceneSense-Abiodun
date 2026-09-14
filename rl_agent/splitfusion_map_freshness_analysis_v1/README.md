# SplitFusion map-freshness policy analysis v1

This package converts the completed 72-action by four-network-profile campaign
into a controller-facing map-freshness surface. It is an offline,
hash-verified counterfactual analysis; it does not replace the measured
campaign and does not claim a second live experiment.

The analysis changes only edge queue discipline while preserving the measured
per-cell feature arrival/admission surface and the qualified final edge-service
calibration:

- `FIFO_NO_DISCARD` drains every admitted frame in arrival order.
- `LATEST_ONLY_NO_EXPIRY` keeps the newest pending frame while active work is
  non-preemptive. Superseded pending frames receive explicit terminal reasons.
  A sole pending frame is never discarded merely because it waited.

Freshness is evaluated at 150, 200 and 250 ms. The primary outcome is the
fraction of route time for which the newest installed map contribution remains
inside the selected age budget. The outputs also retain timely useful-update
yield, empirical queue/AoI distributions, frozen person/vehicle quality,
quality-weighted freshness, fixed-action network sensitivity, and the limited
aligned-localization evidence available from the campaign.

Run from the repository root:

```bash
python3 -u -m rl_agent.splitfusion_map_freshness_analysis_v1.analyze_288 \
  --output /create-only/output/path

python3 -u -m rl_agent.splitfusion_map_freshness_analysis_v1.plot_results \
  --analysis-root /create-only/output/path
```

Focused CPU checks:

```bash
python3 rl_agent/splitfusion_map_freshness_analysis_v1/test_queue_models.py
python3 rl_agent/splitfusion_map_freshness_analysis_v1/test_analysis.py
```

## Capture clock versus action clock

The follow-up analysis preserves capture-to-install AoI and adds a second,
controller-facing boundary at `capture_started_ns`, the entry to seven-channel
tensor assembly immediately before action dispatch. It uses the paired UE
monotonic/wall timestamps recorded at result reception as a measured clock
bridge; it does not subtract a fixed sensor-preparation constant.

It also corrects one causal weakness in the earlier offline arrival imputation:
an imputed edge arrival may not precede that frame's measured UE transmission
start. Observed edge arrivals are never changed. The earlier results remain in
the output as an explicit comparison.

```bash
python3 -u -m rl_agent.splitfusion_map_freshness_analysis_v1.analyze_timing_boundaries \
  --output /create-only/output/path

python3 -u -m rl_agent.splitfusion_map_freshness_analysis_v1.plot_timing_boundaries \
  --analysis-root /create-only/output/path

python3 rl_agent/splitfusion_map_freshness_analysis_v1/test_timing_boundaries.py
```

Physical capture-clock AoI remains the safety and reward quantity. The action
clock is for attribution, diagnosis, and policy state: the agent should observe
how old its prepared input already is rather than be penalized as though it
controlled upstream CARLA/sensor synchronization.

The current FCOS `LOCAL_INFER` branch is intentionally not synthesized from
older LR-ASPP measurements. It requires a separately frozen and measured local
compute, compact-result, quality, transport, installation and ACK contract
before it may enter the action set.
