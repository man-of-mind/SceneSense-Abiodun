# Presenter notes — UE split-inference supervisor baseline

The deck content is fully editable. The notes below are also embedded in each slide's speaker notes.

## Slide 1 — UE Split-Inference Baseline

This is deliberately a measurement plan, not an RL result. The central question is whether the network conditions create a meaningful profile-selection problem at all. If they do not, the correct outcome is a simple rule rather than forcing a learning result.

## Slide 2 — Start with the smallest question the UE agent must answer

Keep the scope narrow: one UE, split inference, and one object-map endpoint. We reuse the same sensor samples and hold compute fixed. SKIP, LOCAL, urgency, cooperation, occlusion reasoning, and RL are later stages and do not block this baseline.

## Slide 3 — Three normal profiles plus one explicit rescue span the trade-off

These four actions are an experimental shortlist from 72 offline configurations. The rescue action is separate because its pedestrian recall is below the normal floor. Segmentation is still measured, but the current service prioritizes object class and world location.

## Slide 4 — The current regimes may not force profile switching

The chart is the most important reality check. All four action loads are below the central historical poor-link capacity. Only Quality under Poor is close enough to the uncertainty band to justify a direct boundary measurement. We should not assume switching or RL is necessary.

## Slide 5 — Define 16 logical cells; directly measure only two before expanding

The 16 cells are a planning surface, not 16 experiments. We initially measure Quality under Clear and Poor. We step down only if Poor fails or is borderline. Inferred feasibility never fills unmeasured latency, drop, map-update, or AoI values.

## Slide 6 — Fixed 10-Hz replay links processing, delivery, and freshness

Every direct cell uses the same 10-Hz replay and fixed pipeline. Freshness starts from the source release time of the newest accepted map update, not from send or enqueue. We report multiple latency summaries and derive the acceptable AoI from the measured error trade-off.

## Slide 7 — The baseline tells us whether learning is necessary

The baseline has three scientifically useful outcomes: a fixed/greedy rule, a measured profile boundary, or evidence that temporal history matters. Only the third outcome motivates a sequential learned policy. Please confirm the scope, shortlist, regimes, two-cell start, and AoI approach.
