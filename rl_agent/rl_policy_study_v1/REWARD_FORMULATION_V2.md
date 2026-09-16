# Split-only PPO reward formulation v2

Status: **ADVISOR-DISCUSSION DRAFT — DO NOT START TRAINING UNTIL THE LATENCY BOUNDARY AND WEIGHTS ARE LOCKED**

## Design decision

The first policy should optimize three interpretable outcomes for each selected
split action:

1. semantic-segmentation quality;
2. physical localization quality; and
3. sensor-compute-start to model-tail-feedback latency.

The returned frame ID is essential, but it is not itself a numerical reward.
It binds delayed feedback to the exact frame and action that caused it, and it
lets the next observation express how many newer frames have appeared since
that action. The first training phase deliberately ends at model-tail feedback;
map installation remains a separately recorded outcome for a later map-aware
extension.

## Candidate 9-FPS service contract

Use 9 FPS as the first **experimental operating point**, not as an assumption
that delayed feedback has disappeared. Its nominal frame period is

$$
P_9=1000/9=111.1\ \mathrm{ms}.
$$

With approximately 30 ms of optimized sensor computation, the next action is
nominally ready about 141.1 ms after the preceding frame's sensor-compute start.
The candidate feedback budget is therefore

$$
B_T=140\ \mathrm{ms}.
$$

The deployable latency variable must end when the compact progress feedback is
**received by the UE agent**:

$$
T_j^{\mathrm{fb}}
=t_{j,\mathrm{TAIL\_COMPLETED\ ACK\ received\ at\ UE}}
 -t_{j,\mathrm{sensor\ compute\ start}}.
$$

The split action is selected after sensor preparation, so it cannot change the
elapsed sensor cost. It can still respond fairly: the observation includes the
measured current-frame sensor-compute time and remaining budget
$B_T-T_j^{\mathrm{sensor}}$. A slow preparation therefore asks the policy for a
more conservative communication action rather than becoming hidden reward
noise.

The present counterfactual analysis ends at model-tail completion on the edge,
so it omits the compact ACK return. Its 140-ms feasibility counts are therefore
provisional upper bounds: 18/72 actions in Favorable Stable, 15/72 in Mid
Variable, 11/72 in Adverse Stable and 14/72 in Fade Recovery have at least 100
timing samples and a P50 at or below 140 ms. No supported action has P95 at or
below 140 ms. This is still a learnable action-selection problem—some actions
are useful deadline candidates—but it is not a synchronous-feedback guarantee.

The evidence-qualified P50-feasible sets contain 18, 15, 11 and 14 actions in
Favorable, Mid, Adverse and Fade. Eleven actions occur in all four sets:
`35, 41, 47, 52, 53, 58, 59, 64, 65, 70, 71`. This is sufficient diversity for
the policy to learn a quality/deadline trade-off without pretending every
valid action is fast. For example, action 52 has the highest joint quality in
this common subset (about 0.402) but reaches 139.6 ms P50 in Adverse before ACK
return; action 70 gives slightly lower quality (about 0.396) with a stronger
worst-profile P50 margin at 127.9 ms. These are distributional candidates, not
guarantees: no action has P95 at or below 140 ms in the current replay.

The future `LOCAL` trigger must not be “most of the 72 split actions look bad.”
Seventy bad choices do not matter if one or two high-utility split choices are
reliably feasible. A later hysteretic gate should compare the **best predicted
feasible SPLIT utility/on-time-useful probability** with independently measured
`LOCAL` latency, quality, energy and result-upload cost. Until those local
measurements exist, `LOCAL` remains a shadow recommendation and is not a
trainable or executable v1 action.

Crossing 140 ms is a service-deadline miss, not automatic proof of radio loss.
If no feedback has arrived at the deadline, the ticket becomes
`DEADLINE_MISSED_PENDING`; a later event must still distinguish late success,
transport failure and feedback loss.

## Normalized quality terms

For a terminal result belonging to frame $j$, let

$$
S_j=mIoU_{\mathrm{seg},j}\in[0,1]
$$

be the aggregate semantic-segmentation quality. The retained evidence does not
provide separate vehicle/person semantic mIoU, so object-overlap metrics must
not be relabelled as class-specific segmentation IoU.

The current localization coordinate combines box/footprint overlap with
centroid-position accuracy:

$$
Q_{\mathrm{overlap},j}
=\sqrt{\mathrm{IoU}_{\mathrm{vehicle},j}
       \mathrm{IoU}_{\mathrm{person},j}},
$$

$$
e_{xy,j}
=\sqrt{\frac{e_{\mathrm{vehicle},j}^{2}
                   +e_{\mathrm{person},j}^{2}}{2}},
\qquad
Q_{xy,j}=\exp\!\left(-\frac{e_{xy,j}}{d_0}\right),
$$

$$
G_j=\sqrt{Q_{\mathrm{overlap},j}Q_{xy,j}}\in[0,1].
$$

Here $d_0=1\,\mathrm m$ is a presentation normalizer, not a correctness gate.
The geometric means are conservative: excellent vehicle performance cannot
hide failed person localization, and strong overlap cannot hide a large
centroid error.

## Latency and deadline penalty

For the first policy, use $T_j^{\mathrm{fb}}$ from the candidate service
contract above. Waiting for CARLA to produce a synchronized sample and visual
rendering are outside it; optimized camera/radar computation is inside it. The
compact feedback return is also inside it because the next action cannot use a
message that has reached only the edge.

Use a bounded smooth penalty:

$$
\phi_T(T_j^{\mathrm{fb}})=1-\exp\!\left(
-\frac{T_j^{\mathrm{fb}}}{\tau_T}\right)\in[0,1].
$$

It is nearly linear for small latency, grows monotonically, and saturates so a
single extreme delay cannot dominate an entire PPO update. The scale $\tau_T$
is a registered service tolerance, not a value selected after training.

Add an explicit, latched deadline event

$$
D_j^{\mathrm{miss}}(B_T)
=\mathbf{1}[\text{no terminal service event for ticket }j
\text{ has reached the UE by }B_T].
$$

This definition also covers failures and intentional supersessions, for which
$T_j^{\mathrm{fb}}$ as a successful-tail latency may not exist. The bit is
latched at $B_T$, booked once when the ticket is reconciled, and remains one if
a late terminal event later arrives. The smooth term ranks successful actions;
the binary term ensures every strict service-budget violation receives an
additional registered penalty.

## Recommended first reward

Let $\bar S_{a_j}$ and $\bar G_{a_j}$ be the frozen validation-quality anchors
for the selected action. They describe expected action quality, not live
per-frame ground truth. For the transition ticket created by frame $j$:

$$
\begin{aligned}
r_j={}&\mathbf{1}_{\mathrm{tail\ completed}}
\left[
w_S\bar S_{a_j}+w_G\bar G_{a_j}
-w_T\phi_T(T_j^{\mathrm{fb}})
\right]\\
&-\beta_T D_j^{\mathrm{miss}}(B_T)
-\beta_D\mathbf{1}_{\mathrm{proven\ transport\ failure}}
-\lambda_A\mathbf{1}[a_j\ne a_{j-1}].
\end{aligned}
$$

Use $w_S,w_G,w_T\ge0$ with $w_S+w_G+w_T=1$ during the first registered weight
sweep. This makes the accuracy/latency trade-off readable. A safety-oriented
study may give localization more weight than aggregate segmentation, but the
actual weights must be agreed before training rather than selected from the
best-looking policy result.

| Symbol | Meaning |
|---|---|
| $\bar S_{a_j}$ | Frozen segmentation-quality anchor for the selected action. |
| $\bar G_{a_j}$ | Frozen localization-quality anchor for the selected action. |
| $T_j^{\mathrm{fb}}$ | Sensor-compute-start to compact tail-feedback receipt at the UE for this exact frame. |
| $B_T$ | Candidate 140-ms service deadline for the first 9-FPS experiment. |
| $D_j^{\mathrm{miss}}(B_T)$ | Latched once if no terminal service event reaches the UE by the budget; it does not claim radio loss. |
| $B_j/B_{\max}$ | Separate byte-cost target; $B_{\max}$ is a fixed catalog normalizer. |
| $C_j/C_{\max}$ | Separate compute-cost target, including work spent before supersession. |
| $\mathbf{1}_{\mathrm{tail\ completed}}$ | One only when exact-identity evidence proves model-tail completion. |
| $\mathbf{1}_{\mathrm{proven\ transport\ failure}}$ | One only after incomplete delivery is proven; silence at the next frame is still `PENDING`. |
| $\mathbf{1}[a_j\ne a_{j-1}]$ | Optional action-switch penalty. |

The v1 accounting mode is explicit: bytes and compute are the two named cost
critics and their scalar reward coefficients are zero. Deadline miss, proven
transport failure and optional action switching remain scalar penalties. Any
later scalar-only ablation or movement between reward and constrained costs
must be separately registered so no event is optimized twice. The numerical
weights in the code are candidates for sensitivity tests, not frozen training
hyperparameters.

## Tail feedback and optional later quality correction

The edge-to-UE wire message should carry at least

$$
\mathrm{TAIL\_COMPLETED\ ACK}
(k_j,t_{j,\mathrm{edge\ tail}},B_j,C_j,
\bar S_{a_j},\bar G_{a_j},
\mathrm{quality\ source/version/catalog\ hash}).
$$

The message cannot know its own future UE receipt time. On receipt, the UE
stamps $t_{j,\mathrm{recv}}$, computes $T_j^{\mathrm{fb}}$ and the deadline
result in one UE clock domain, and forms the enriched terminal record. Thus the
next observation can identify exactly which prior frame completed, how long it
took, whether it met 140 ms, and what validated quality level that action
represents. The quality fields are not claimed to be realized accuracy
for the current live frame. True per-frame accuracy requires ground truth and,
for localization, later post-processing; ordinary deployment has neither at
this instant.

In CARLA or another labelled environment, a later evaluator may issue an
exact-ticket correction

$$
\Delta r_j^Q=\mathbf{1}_{\mathrm{tail\ completed}}
\left[w_S(S_j-\bar S_{a_j})+w_G(G_j-\bar G_{a_j})\right].
$$

That correction must be reconciled before the affected PPO rollout prefix is
updated; standard on-policy PPO cannot update once and then retroactively edit
the transition. The first trace-driven policy may instead use only the frozen
quality surface and close its control ticket at tail feedback. `MAP_OUTCOME`
remains valuable for auditing and for a later map-aware policy, but it is not
required to release this first tail-service reward. No quality, latency or
resource component may be booked twice.

## Frame identity and delayed feedback

Every action opens a ticket keyed by

$$
k_j=(\mathrm{session\_id},\mathrm{ue\_id},\mathrm{frame\_id},\mathrm{action\_id}).
$$

The UE-enriched terminal record—not the raw edge wire message—should contain

$$
(k_j,\;\mathrm{outcome},\;\bar S_{a_j},\;\bar G_{a_j},\;
T_j^{\mathrm{fb}},\;B_T,\;D_j^{\mathrm{miss}},\;
\mathrm{quality\ source/version/catalog\ hash}).
$$

If feedback for frame $j$ has not arrived when frame $j+1$ is ready, ticket
$j$ remains `PENDING`; the agent chooses again using pending age/count, BSR,
the latest tail-feedback frame lag. At 140 ms it may
be marked `DEADLINE_MISSED_PENDING`, but it must not invent a radio-loss
outcome. Out-of-order feedback closes the exact ticket, not merely the most
recent action.

Frame identities yield causal tail-service observation features such as

$$
\Delta f_t^{\mathrm{tail}}=f_t-f_{\mathrm{latest\ tail\ feedback}},
\qquad
\Delta f^{\mathrm{pending}}_t
=f_t-f_{\mathrm{oldest\ pending}}.
$$

At each decision cutoff, the observation drains all received exact-identity
events and exposes the most recent quality anchor/proxy, feedback latency,
deadline result, feedback age, frame lag, pending count/age and rolling on-time
statistics. Raw frame ID is not fed to the network as an ever-growing scalar:
that would be episode-dependent and scientifically meaningless.

## Terminal-outcome rules

- `TAIL_COMPLETED_ACK_RECEIVED`: close the first-phase service ticket using the
  frozen quality anchor, received feedback latency and deadline result.
- `PENDING`: assign no reward yet.
- `DEADLINE_MISSED_PENDING`: record the service miss for the next observation,
  but do not call it transport loss; reconcile the ticket later.
- `SUPERSEDED_PENDING`: no fictitious quality and no transport-loss penalty;
  charge bytes/compute already consumed and report the replacement frame.
- `REASSEMBLY_FAILED`: apply the proven-transport-failure penalty.
- stale-before-edge or execution failure: close with its distinct terminal
  class; never relabel it as radio loss.
- `MAP_OUTCOME`: retain as a separate audit/evaluation event until a later
  map-aware reward version is explicitly registered.
- missing feedback: reconcile against the durable edge ledger before expiry.

## Evidence and deployment boundary

During first-phase trace-driven training, $\bar S_{a_j}$ and $\bar G_{a_j}$
come from the frozen 72-action validation surface and receive credit only when
tail completion is proven. CARLA ground truth is not available to a real
deployed vehicle, so live per-frame adaptation requires a separately qualified
quality proxy; otherwise deployment uses the fixed offline action-quality
anchors. BSR, SNR and MCS are causal observations; they help predict delivery
and latency but are not themselves rewards.

Before training begins, lock: the 9-FPS cadence, received-ACK timestamps,
$B_T$, $\tau_T$, the weight grid, terminal reconciliation horizon, whether
deadline/resources are scalar penalties or constrained costs, and the
treatment of metrics unavailable at deployment. The compact early-ACK return
must be measured live before claiming the 140-ms contract is deployable.
