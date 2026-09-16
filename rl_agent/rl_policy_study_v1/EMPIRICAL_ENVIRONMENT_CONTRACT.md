# SplitFusion empirical RL environment contract v1

Status: **9-FPS / 140-MS CANDIDATE CONTRACT — REQUIRES EARLY-ACK LIVE VALIDATION**

## 1. Decision being learned

At each nominal 111.1 ms decision epoch (9 FPS), the first agent selects exactly
one of the 72 registered `SPLIT` profiles:

$$
a_t \in \{0,\ldots,71\}.
$$

Each action fixes the model family, quantizer, and one measured $q$ anchor.
`LOCAL` and `SKIP` are outside this first training contract. They can be added
only after their own causal transition and cost measurements exist.

All 72 scientifically valid but poor actions remain available. A hard action
mask is permitted only for an execution fact such as a missing resident model,
invalid codec binding, failed device allocation, or protocol-integrity error.

## 2. Why this is a POMDP

The UE observes delayed radio telemetry and past results, not the latent future
channel or the complete edge state. The policy therefore receives only causal
observations $o_t$ and maintains an LSTM memory state:

$$
(h_t,c_t)=\operatorname{LSTM}_\theta(o_t,h_{t-1},c_{t-1}).
$$

The target initial observation contract contains normalized values plus an
explicit availability bit for every telemetry group:

- current and lagged PUSCH SNR, MCS, delivered throughput, and recent radio
  delivery outcomes;
- current-frame sensor-compute elapsed time and remaining 140-ms service budget,
  which are causal because the split action is chosen after sensor preparation;
- latest tail-feedback frame lag, feedback age, segmentation/localization
  quality anchor or deployable proxy plus provenance/availability, received
  latency divided by 140 ms, deadline result, oldest-pending-frame lag,
  pending age/count and availability flags;
- edge busy/pending state and the most recent explicit terminal outcome;
- previous action (categorical, never treated as an ordinal number), payload,
  and tail-feedback latency/deadline outcome; and
- causal ego/radar risk summaries that are available before action selection.

The authored network-profile name, future SNR, ground truth, and the current
frame's tail output are forbidden policy inputs.

`observation.py` currently implements the versioned minimal feedback-path
adapter: current radio signals, current sensor budget, categorical previous and
feedback action identities, exact cutoff-safe tail/pending state, terminal
history and availability bits. Temporal radio history enters through repeated
LSTM steps rather than an authored profile label. The compact causal
scene/risk summary is not yet implemented, so this is not the final frozen
training schema and training remains gated on that definition.

## 3. Explicit causal channel forecast

The supervisor's forecasting suggestion is implemented as a causal policy
input plus a supervised forecasting objective, not as access to future
information. Do not conflate the 140-ms end-to-end feedback deadline with the
channel-forecast horizon. For v1, let $H_{\mathrm{ch}}=100$ ms and let
$y_{t,H_{\mathrm{ch}}}$ be the normalized **mean SNR in the next registered
radio-exposure window** $(t,t+H_{\mathrm{ch}}]$. The window begins strictly
after action selection and covers the period in which the chosen feature is
expected to traverse the uplink. The forecast head reads the current LSTM
state and predicts a Gaussian location and positive predictive scale:

$$
(\widehat\mu_{t,H_{\mathrm{ch}}},\widehat\sigma_{t,H_{\mathrm{ch}}})
=g_\psi(h_t),
\qquad \widehat\sigma_{t,H_{\mathrm{ch}}}>0.
$$

The acting policy receives the prediction in the same forward pass:

$$
\pi(a_t\mid o_{\le t})=
\operatorname{softmax}\!\left(
W_\pi[\,h_t;\operatorname{sg}(\widehat\mu_{t,H_{\mathrm{ch}}},
\log\widehat\sigma_{t,H_{\mathrm{ch}}})\,]+b_\pi
\right).
$$

Here `sg` is stop-gradient. It blocks the direct policy-loss path into the
forecast-head parameters, while the encoder and LSTM remain a shared trunk and
therefore remain coupled to both objectives. The prediction is allowed at
decision time because it depends only on $o_{\le t}$; $y_{t,H_{\mathrm{ch}}}$ is constructed
only after the future window closes and is never a policy input.

The v1 forecast loss is Gaussian negative log likelihood up to its additive
constant:

$$
L_{\mathrm{forecast}}=
\mathbb E_t\!\left[
\frac{1}{2}\left(
\frac{y_{t,H_{\mathrm{ch}}}-\widehat\mu_{t,H_{\mathrm{ch}}}}
{\widehat\sigma_{t,H_{\mathrm{ch}}}}
\right)^2+
\log\widehat\sigma_{t,H_{\mathrm{ch}}}
\right].
$$

Windows without a valid later SNR observation are excluded from this loss
rather than filled from future or authored profile information.

The combined optimization objective is

$$
L=L_{\mathrm{PPO}}+c_VL_V+c_FL_{\mathrm{forecast}}-c_H\mathcal{H}(\pi).
$$

The forecast head regularizes the recurrent representation and gives the policy
an explicit short-horizon channel belief. Both predicted mean and uncertainty
are supplied to the policy head; the true future target may never be supplied.
The predicted scale is called uncertainty, not calibrated uncertainty, until
held-out likelihood and empirical-coverage checks support that claim. The first
ablation compares no forecast, an auxiliary-only forecast and the explicit
detached forecast input. Rollout rows store the exact detached forecast pair
used by the acting policy. PPO re-evaluation reuses that pair rather than
recomputing it after a forecast-head update; parameters are frozen during
rollout and actual KL is monitored during optimization.

## 4. Transition and feedback semantics

Each transmitted frame opens a pending ticket keyed by session, UE, frame and
action identity. If the next frame arrives first, the previous ticket remains
`PENDING`; it is not inferred to be lost. At 140 ms an unresolved ticket becomes
`DEADLINE_MISSED_PENDING`: this is a service miss but not proof of radio loss.
The policy may choose the next action using pending age/count and tail-feedback
lag. Delayed or out-of-order feedback closes the exact ticket it names.

The first experiment registers `tail_only_v1`. The edge sends a compact
`TAIL_COMPLETED_ACK` carrying exact identity, its completion event, realized
resource charges and quality anchors/proxy with source/version/catalog hash.
The wire message cannot know its future UE-received latency. The UE stamps its
arrival, derives the same-clock feedback latency and deadline result, and then
closes the PPO service ticket with that enriched record.
`MAP_OUTCOME` is retained independently and cannot retroactively mutate an
already-consumed on-policy transition. Each service ticket ultimately receives
exactly one terminal outcome. The learning environment keeps these classes
separate:

- model-tail feedback received;
- intentional supersession in the compute or publication pending slot;
- queue-wait expiry;
- processing-horizon expiry at a named stage;
- measured transport-incomplete or pre-queue rejection; and
- structural/runtime integrity failure.

Intentional supersession is not relabelled as radio loss. It earns no new-map
utility, but it remains charged for feature bytes and compute already spent.
Structural faults terminate an episode and are not ordinary negative samples.
Missing feedback may expire only after a cumulative terminal watermark or a
durable edge-ledger reconciliation proves that no terminal record exists.
Duplicate identical feedback is idempotent; an identity conflict is structural.

PPO rollout rows retain the old log probability, value estimate and incoming
LSTM state and detached forecast features when the ticket opens.
Each UE/session has its own ordered ledger. A rollout uses one fixed actor
version and a declared terminal horizon; generalized-advantage estimation and
updates wait until every ticket through that horizon is reconciled. An
arbitrary contiguous prefix is not update authority if an already-collected
suffix would become stale. This bookkeeping does not replace PPO
ratio/clipping or measured KL.

## 5. Initial 140-ms tail-service reward

The first implementation does not feed an unbounded raw frame number to the
network or make frame ID an arbitrary scalar reward. Frame identity provides
credit attribution and derives the causal tail-feedback lag

$$
\Delta f_t^{\mathrm{tail}}=f_t-f_{\mathrm{latest\ tail\ feedback}}.
$$

For the action ticket belonging to frame $j$, the initial reward is

$$
\begin{aligned}
r_j={}&\mathbf{1}_{\mathrm{tail\ completed}}
\left[w_S\bar S_{a_j}+w_G\bar G_{a_j}
-w_T\phi_T(T_j^{\mathrm{fb}})\right]\\
&-\beta_TD_j^{\mathrm{miss}}
-\beta_D\mathbf{1}_{\mathrm{proven\ transport\ failure}}
-\lambda_A\mathbf{1}[a_j\ne a_{j-1}].
\end{aligned}
$$

$T_j^{\mathrm{fb}}$ starts at production sensor-compute start and ends when the
compact tail feedback is received by the UE. The frozen $\bar S$ and $\bar G$
values are action-quality anchors, not claimed live per-frame ground truth.
The smooth term ranks successful actions. $D_j^{\mathrm{miss}}$ latches once
when no terminal service event has reached the UE by 140 ms, regardless of the
eventual outcome, and is charged once after reconciliation.
`SUPERSEDED_PENDING` earns no fictitious quality and no radio-loss penalty.
Bytes and compute already spent are returned as the two named cost-critic
targets, not subtracted again from this scalar reward.

### Deferred object-level refinement

If later application evidence justifies per-object criticality and freshness
tolerances, let $\mathcal{O}_t$ be the tracked objects in the map, $q_{i,t}$ an
auditable task-quality score, $A_{i,t}$ object AoI, $w_i$ criticality, and
$\tau_i$ a registered freshness constant:

$$
U_t=
\frac{\sum_{i\in\mathcal{O}_t}
w_i q_{i,t}\exp(-A_{i,t}/\tau_i)}
{\sum_{i\in\mathcal{O}_t}w_i+\varepsilon}.
$$

A later reward may use $U_{t+1}-U_t$, but this richer form is deferred and is
not silently combined with the initial latency discount. Reward weights and
freshness constants require a normalization and sensitivity study on
training-only transitions. Safety or resource requirements that must hold are
reported as separate cost signals rather than hidden inside one reward number.

## 6. Evidence that can and cannot train this environment

The completed 288-cell live campaign is authoritative for the original runtime
and provides all 72 actions under all four network processes. The optimized
latest-only direct-map 288-cell counterfactual preserves measured per-cell
reassembly and admission totals, removes the erroneous edge→UE→map detour, and
recomputes map outcomes after the edge optimization. Compact record-free
feedback returns separately to the UE and is excluded from physical
map-installation age. The current early-boundary replay ends at edge model-tail
completion; its compact feedback return is still unmeasured and must not be
silently treated as zero.

It is suitable for:

- action-scale normalization;
- initializing action-conditioned outcome models;
- checking payload/quality/freshness reward sensitivity; and
- rejecting obviously inconsistent simulator behavior.

It is not by itself a sequential switching trajectory. In particular, 258,138
of 602,315 edge-admission timestamps are deterministically imputed because the
original campaign retained aggregate counters but not the identities of every
superseded arrival. PPO must not train directly on those rows as though every
timestamp were observed.

## 7. Required implementation gate before PPO

Build a causal trace-driven environment that:

1. aligns each frame only with radio telemetry available at its decision time;
2. evolves the depth-one two-stage edge scheduler across action changes;
3. reports observed versus imputed transition components;
4. reproduces exact tail-feedback tickets, 140-ms miss rate and explicit
   terminal outcomes, while retaining map outcomes separately;
5. withholds authored profile identity and future SNR from the policy;
6. separates training traces/seeds from evaluation traces/seeds; and
7. reproduces measured fixed-action cell statistics within preregistered
   tolerances before any PPO result is accepted.

Only after that reproduction gate passes should the LSTM-PPO policy and its
forecast-head ablation be trained.
