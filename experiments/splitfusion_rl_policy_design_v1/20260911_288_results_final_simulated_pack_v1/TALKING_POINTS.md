# Final 288-cell presentation talking points

## What changed

The original campaign remains immutable. The final-edge simulator preserves
the measured payload, radio reassembly and edge-admission outcomes, then
causally replays the depth-one latest-only scheduler with the final edge
calibration. Therefore simulated installation and AoI are counterfactual—not
new live measurements.

Overall installed/sent rises from 0.373
to 0.647. Median cell capture-to-install AoI
falls from 364.5 ms to
235.8 ms. Time-weighted map AoI falls from
573.5 ms to
307.2 ms.

## Figures 01–02

The validation and localization scores are frozen action-level measurements;
they were not measured four times and are not medians across network profiles.
Each column now uses that profile's measured payload. Marker size represents
the simulated installation probability, which is the network-dependent part.
The same intrinsically good action can therefore be useful under favorable
conditions but rarely installed under adverse conditions.

## Figure 03

Unchanged: q controls the intrinsic payload/quality tradeoff. Solid lines are
payload; dashed lines are person F1.

## Figure 04

This is simulated installed/sent after final edge optimization. The profile
rates are Favorable stable 0.758, Mid variable 0.651, Adverse stable 0.511, Fade recovery 0.668.
The numerator is an authoritative simulated map installation; the denominator
is every measured UE-sent feature.

## Figure 05

Unchanged measured radio evidence. UDP does not retransmit. Datagram reception
means datagrams observed at the edge divided by datagrams sent; complete
reassembly requires every fragment of a feature message.

## Figures 06–07

The simulator expands the feasible action region, but the network profile still
matters. This is precisely why the agent should condition decisions on causal
channel and recent-delivery history rather than select one globally fixed
action.

## Figure 08

Shows simulated capture-to-authoritative-install AoI only for actions with at
least one installation in that profile. The former 100-ms line was removed as
requested. It remains a separately reportable reference, not the optimization
or admission horizon.

## Figure 09

Unchanged original live four-action breakdown. It explains why edge service
was optimized, but its component medians are descriptive and are not an exact
additive reconstruction of Figure 08.

## Figure 10

Compares only the original edge processing with the final target. A30/A50/A71
use final v3. NoAE uses the last live-valid v2 because action 15 failed v3
closed on non-finite camera-aware geometry. The plot deliberately omits
intermediate implementations. V3 overlaps GPU/CPU stages, so component times
cannot be presented as an additive stack; worker-start-to-publication is the
scientifically valid before/after total.

## Agent transition

The policy diagram connects the evidence to split-only recurrent PPO. The
action determines intrinsic quality and payload; the channel and edge state
determine whether the update arrives and remains useful; installed-object AoI
determines freshness utility. The next-SNR head is an auxiliary training task,
not access to future channel information.
