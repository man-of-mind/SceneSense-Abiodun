# Six presentation PNGs

Use `figures_02`. All six are 3840×2160, visually inspected. Colours consistently identify scene (orange), channel (blue), backlog (green-grey) and complete previous decision/outcome (purple). No PDFs. `figures_01` is retained draft layout, not an additional analytical result.

## 1. [Four-condition summary](figures_02/01_four_condition_summary.png)

For 100 uniformly spaced decisions in each full Town10 map capture, bars show mean absolute grouped Shapley contributions relative to the same eight-state background. Mode-11 preference and conditional q are separate outputs; black ticks use the single predeclared alternative background. Scene is largest under both backgrounds, but magnitudes and secondary rankings are reference-sensitive. Far percentages use 100 × 8 × 14 nonendpoint coalition slots per run. These are not causal percentages, training-distribution importance, or evidence of better quality/latency; every target has at least one far hybrid in its background decomposition.

## 2. [M3.1 GOOD_ALL timeline](figures_02/02_M3_1_timeline.png)

All 1,771 observed decisions show categorical mode, executed q, observed payload, channel/scene/backlog and history context. Bars explain changes from the immediately preceding decision at 99 of the 100 uniform targets (the first has no predecessor). Even with commanded SNR 23.5 dB and steady MCS, changing history and scene move actor outputs. Gaps are unexplained times, not zero effects. No interpolation of attributions or causal payload explanation is implied; diamonds show unexecuted q11 only at explained non-11 choices.

## 3. [M3.2 BAD_ALL timeline](figures_02/03_M3_2_timeline.png)

All 1,550 decisions under commanded 8.5 dB are aligned with 99 sampled preceding-decision decompositions. Timeout history predominates; scene, history and intermittent backlog vary despite constant commanded SNR. Red crosses retain and flag far coalitions. A zero channel contribution on these sampled steps does not mean channel inputs are irrelevant. Payload is observed SFD4 kB, not an attributed output, and measured PUSCH is neither plotted nor substituted for the commanded input.

## 4. [M3.3 GOOD_TO_BAD timeline](figures_02/04_M3_3_timeline.png)

The 1,533-decision full Town10 map timeline includes a separate 60-decision transition view (20 before, first 20 after, following 20), aligned at the first new-SNR decision. q initially falls, then later cleared-history intervals accompany higher q. Signed bars separate the channel and history contributions to the first change. The dense transition cohort does not enter the uniform summary unless a target was already selected there. Far hybrids limit interpretation; V13's original registered STOP remains unchanged.

## 5. [M3.4 BAD_TO_GOOD timeline](figures_02/05_M3_4_timeline.png)

All 1,529 decisions and a separate 60-decision transition view show the first GOOD decision coinciding with cleared previous history. The q rise is allocated mainly to history on that step, followed by a decline when defined history returns. This contrasts with M3.3's immediate decrease. The first-step decomposition includes far hybrids and does not isolate missing quality from the other history fields. Commanded SNR and prior MCS are actor inputs; this is not measured achieved SNR or evidence of a beneficial live intervention.

## 6. [Waterfall examples](figures_02/06_waterfall_examples.png)

Six retained principled cases cover constant GOOD, constant BAD, a mode switch, and both transitions. Each row has one preceding→target contrast and all four group contributions, in logit-margin and q-fraction units. The GOOD→BAD q decrease and BAD→GOOD/reset increase have different allocations. Red titles flag far counts out of 14 nonendpoint hybrids; no coalition is removed. At the actual 9→8 switch, the figure explains mode 8's margin and q8, not a continuous mode number: q8 rises although executed q falls. These cases are illustrative, not population importance or payload/quality/latency predictions.
