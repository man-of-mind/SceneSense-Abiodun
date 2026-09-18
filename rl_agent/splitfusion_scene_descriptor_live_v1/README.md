# Live SI/P40 sensor-path validation

This package measures the cost of the frozen Hybrid-SAC scene descriptors in
the qualified live action-50 sensor path. It does not run or authorize another
288-cell campaign.

The A/B contract is fixed to two fresh `FAVORABLE_STABLE` cells:

- descriptor off (`OFF`);
- descriptor on (`SI_P40_V1`).

Each cell stops intentionally at exactly 520 successful feature
transmissions. The first 20 sent frames are warm-up and the following 500 are
the analysis population. Semantic/object evaluation queues remain enabled and
are drained; full Route-B completion is neither required nor claimed.

Camera SI is computed from the already-resized 768x448 RGB image, before
tensor packing. P40 is computed from raw radar provenance belonging only to
the current non-overlapping 100-ms sweep (`sweep_offset == 0`) after the same
finite, positive, at-most-120-m validity mask already used by the production
adapter. Raw, valid and rejected counts remain separately observable. Both
operations are read-only, and live equivalence gates require the radar evidence
and seven-channel model input to remain exactly equal to production.

Offline preflight:

```bash
python3 -u -m rl_agent.splitfusion_scene_descriptor_live_v1.live_ab --preflight
```

Live A/B:

```bash
python3 -u -m rl_agent.splitfusion_scene_descriptor_live_v1.live_ab \
  --execute SPLITFUSION_SCENE_DESCRIPTOR_LIVE_AB_V1_EXECUTE \
  --output-root experiments/splitfusion_scene_descriptor_live_v1/20260917_action50_favorable_ab
```
