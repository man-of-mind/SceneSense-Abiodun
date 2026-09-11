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

## Figures 01a–01d and 02a–02d

The validation and localization scores are frozen action-level measurements;
they were not measured four times and are not medians across network profiles.
Each network profile now has its own readable page: three quality panels in
Figure 01 and four localization panels in Figure 02. Each page uses that
profile's measured payload. Marker size represents the simulated installation
probability, which is the network-dependent part. The same intrinsically good
action can therefore be useful under favorable conditions but rarely installed
under adverse conditions.

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

Uses the same eight stage labels and colors as Figure 09, with paired `Before`
and `Final` bars for each action. Sensor preparation, UE dispatch and OAI
uplink are held at their original medians because this intervention changed
only the edge path. A30/A50/A71 use final v3; A15 uses its last live-valid v2
because its v3 run failed closed on non-finite camera-aware geometry. The
stage medians are descriptive rather than an additive end-to-end identity;
the annotation beside each final bar reports the directly measured edge-total
saving.

## Agent transition

The architecture document follows the LR-ASPP/FCOS documentation style: a
portable SVG followed by the Mermaid source. The LSTM keeps a compact memory
of recent channel, delivery, map-freshness and action history. Its state feeds
the PPO policy and critics, while a separate auxiliary head learns to forecast
the next observed SNR. That forecast shapes useful temporal features during
training; future SNR is never supplied to the acting policy.
