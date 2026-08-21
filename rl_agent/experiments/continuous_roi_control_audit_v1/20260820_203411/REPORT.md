# Continuous ROI-Drop Control Audit

Audit: `continuous_roi_control_audit_v1/20260820_203411`  
Terminal: **`INSUFFICIENT_EVIDENCE`**  
Controller recommendation: **keep measured discrete q anchors** pending dense-q evidence.

## Answer

The current 72-profile evidence does **not** establish q as a valid continuous
control in any of the 12 fixed family/quantizer branches. It contains only four
in-distribution q anchors (`0,.3,.5,.7`), so it cannot resolve local behavior at
the planned 0.05 scale or score held-out midpoint q values. This is not a finding
that q is intrinsically discrete-only: the preregistered `DISCRETE_ONLY` terminal
also requires a complete dense-q run. The defensible status is
`INSUFFICIENT_EVIDENCE`, with the controller remaining discrete.

## Preflight and stop rule

The frozen plan was written before inference. Preflight then triggered its stop
rule because the recorded dataset manifest/object file are absent and PyTorch
reports CUDA unavailable. No CPU full-run fallback was allowed. Dense inference,
CARLA, and OAI were not started; all results below derive from immutable existing
anchor CSVs.

- Planned grid: 19 q values; 228 profiles; 492,936 profile-frame evaluations.
- Audit validation: 838 unique frames.
- Frozen audit test: 1,324 unique frames.
- Identifier and trajectory-block overlap: zero.

## Production q semantics and piecewise action

Production and evaluator code both use rank drop: independently for the native
low/high feature maps, compute objectness ordering and zero the
`round(q*N)` lowest-ranked cells. q is **not** a score threshold. With 5,184 low
cells and 1,296 high cells, `[0,.8]` has
**5,185 joint mask-count plateaus** separated
by 5,184 transitions. Plateau
widths range from 0.0000965 to
0.0001929 in q. Thus a float-valued API
produces a fine but integer, piecewise-constant actuator. The planned 0.05 grid
tests macro smoothness; it does not prove single-cell smoothness.

## Existing-anchor evidence

All 72 profiles are complete: 2,162 unique frames at six q anchors for every
branch. Aggregate audit-test payload is non-increasing over the measured
in-distribution anchors in **12/12 branches**.
The worst frame-paired non-increasing/tied rate over those large anchor gaps is
**100.00%**. These are coarse payload facts, not continuous
quality validation.

At q=0.7, the 12 separate branch outcomes are:

| family | quantizer | payload_bytes_mean | veh_precision | veh_recall | ped_precision | ped_recall | fp_per_frame | xy_mae_m | xy_rmse_m | miou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ae128 | uint4 | 61054.0619 | 0.4941 | 0.9168 | 0.6180 | 0.8767 | 1.4502 | 0.9590 | 1.2860 | 0.5432 |
| ae128 | uint6 | 112351.9396 | 0.4891 | 0.9148 | 0.6127 | 0.8842 | 1.4811 | 0.9555 | 1.2779 | 0.6575 |
| ae128 | uint8 | 144040.4834 | 0.4904 | 0.9141 | 0.6135 | 0.8820 | 1.4728 | 0.9517 | 1.2714 | 0.6817 |
| ae32 | uint4 | 34746.2183 | 0.4978 | 0.9101 | 0.6181 | 0.8480 | 1.4139 | 0.9317 | 1.2329 | 0.6302 |
| ae32 | uint6 | 63290.8323 | 0.4971 | 0.9115 | 0.6116 | 0.8502 | 1.4298 | 0.9221 | 1.2276 | 0.7434 |
| ae32 | uint8 | 81115.4864 | 0.4958 | 0.9121 | 0.6119 | 0.8512 | 1.4358 | 0.9241 | 1.2301 | 0.7623 |
| ae64 | uint4 | 45626.6035 | 0.4980 | 0.9041 | 0.6418 | 0.8704 | 1.3792 | 0.9424 | 1.2613 | 0.6476 |
| ae64 | uint6 | 82930.2508 | 0.4993 | 0.9068 | 0.6362 | 0.8735 | 1.3867 | 0.9374 | 1.2631 | 0.7584 |
| ae64 | uint8 | 106438.8353 | 0.5007 | 0.9075 | 0.6352 | 0.8714 | 1.3822 | 0.9343 | 1.2581 | 0.7788 |
| noae | uint4 | 188948.9041 | 0.5199 | 0.8715 | 0.6103 | 0.8204 | 1.2855 | 1.1347 | 1.4769 | 0.5865 |
| noae | uint6 | 329760.4237 | 0.5262 | 0.8702 | 0.6315 | 0.8321 | 1.2341 | 1.0684 | 1.3994 | 0.7495 |
| noae | uint8 | 420635.5446 | 0.5271 | 0.8728 | 0.6307 | 0.8385 | 1.2372 | 1.0622 | 1.3952 | 0.7627 |

Object quality is not assumed monotonic. `anchor_curve_diagnostics.csv` records
slope reversals per branch/metric, while `paired_anchor_bootstrap.csv` provides
2,000 trajectory-block paired confidence intervals. The audit found
**117 in-distribution ordering crossings** across family or
quantizer comparisons, reinforcing that branch factors are not safely separable.

## Interpolation and smoothness limit

Only a coarse leave-one-current-anchor-out diagnostic is possible: predict q=.3
from q=0/.5 and q=.5 from q=.3/.7. Its audit-test maximum errors are:

| metric | absolute_error |
| --- | --- |
| fp_per_frame | 0.04773 |
| iou_person | 0.04299 |
| iou_vehicle | 0.08713 |
| miou | 0.03993 |
| payload_bytes_mean | 20232.79985 |
| ped_precision | 0.01306 |
| ped_recall | 0.02423 |
| veh_precision | 0.00749 |
| veh_recall | 0.00719 |
| xy_mae_m | 0.05499 |
| xy_rmse_m | 0.05591 |

Those targets are existing anchors and the gaps are 0.2--0.3 wide. They are not
the preregistered held-out `.05,.15,...,.75` midpoint test and cannot earn a
continuous terminal. No dense local discontinuity test was run. `.9/.98` remain
measured extrapolation references and are excluded from the continuous decision.

## Latency

q-gating and serialization latency is `NOT_MEASURED`. Historical end-to-end or
technical-smoke numbers do not isolate objectness/ranking, mask application,
integrated AE, quantization, serialization, and zstd-3 as required by the frozen
plan, so they were not substituted.

## Hybrid-action implication

Even if q later passes, the action is not plain continuous SAC/TD3: it is a
categorical choice among 12 `{family, quantizer}` branches plus a conditional
bounded `q in [0,.8]`. It needs an explicit hierarchy, parameterized-action
critic, or categorical branch policy with a conditional q actor. This audit adds
no q-selection reward and implements no RL agent.

## Smallest next experiment

Restore the exact hashed dataset and a CUDA device, then run **AE64/uint6 on the
838-frame audit-validation split across
all 19 q values**: 15,922
profile-frame evaluations, reusing backbone features/ranking per frame. Stop
there if midpoint interpolation or local-jump criteria fail.

Before promotion, evidence must expand to all 12 branches on the frozen
1324-frame audit test, pass the preregistered
branch criteria with trajectory-block paired uncertainty, include
production-equivalent gate/serialization timing, and show 12/12 branch support.
Until then: no q promotion, no registry change, and no continuous/hybrid policy
implementation.
