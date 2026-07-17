# Month-2 summary — reconciled through 2026-07-16

**North Star:** learn a network-aware split-inference control policy that cuts
payload/latency while preserving task utility and spatial-map freshness.

**Month-2 exit criterion:** an offline controller can compare static, heuristic,
and learned policies using the same logged metrics.

## Status

- **Static action/model characterization: complete.**
- **Single-UE OAI transport characterization: complete for the current
  no-impairment setup.**
- **Offline controller and controller guardrails: not implemented.** This is
  the remaining Month-2 closure item.

## Delivered

1. **M-prime robust baseline.** Drop-aware training preserves clean
   segmentation/localization while supporting importance-ranked ROI actions.
2. **Integrated resident models.** AE-128, AE-64, AE-32, and no-AE models were
   trained/evaluated. Joint training resolves the object-head collapse seen in
   the earlier standalone AE experiments.
3. **Authoritative action table.** `PERMODEL_KNOB_MATRIX.md` contains 42
   profiles across resident model, quantization 8/6/4, and ROI 0/0.3/0.5.
4. **OAI compression A/B.** No-AE u8 (`1141 KB`, `209 ms`, `75%` delivery)
   versus AE-128 u4 (`142 KB`, `77 ms`, `99%` delivery) demonstrates that
   payload controls queueing and availability in the current OAI RFsim path.
5. **Network-configuration diagnosis.** TDD/5QI changes barely affect the
   single-UE/no-impairment result. Revisit network actions under contention or
   channel impairment, where scheduling priority can matter.
6. **Dynamics-aware requirement.** Object-speed/latency and held-map FPS
   experiments establish that the controller must reason about map age
   `Y + 1/FPS`, not per-frame accuracy alone.

## Current findings

- Integrated AE-128 u4 with ROI off is the strongest measured low-payload OAI
  point. It retains useful task quality while restoring approximately `99%`
  result availability.
- Quantization is inexpensive; aggressive ROI primarily harms segmentation.
  AE+ROI combinations therefore require task-specific guardrails.
- The network does not corrupt a completely reassembled frame; overload costs
  availability and freshness. Staleness is the mechanism that turns delay into
  localization error.
- No static profile is universally optimal across scene dynamics, task floors,
  payload, latency, and availability. This is the measured motivation for an
  adaptive policy.

## Remaining closure work

1. Implement the trace join and route-masked action catalog.
2. Implement the reward scorer with task, payload, latency, availability, and
   `Y + 1/FPS` staleness terms.
3. Compare send-everything, lowest-byte, best-fixed, network-only, task-only,
   and scene+network heuristics.
4. Implement controller-level accepted/clamped/rejected guardrails.
5. Train/evaluate LinUCB and show whether it beats the strongest heuristic.

Controlled impairment, multi-UE contention, per-tensor packet priority, and
full Sionna coupling are Month-3 follow-ons after offline replay passes.
