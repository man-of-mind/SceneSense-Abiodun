# SplitFusion empirical RL environment contract v1

Status: **DESIGN FROZEN FOR IMPLEMENTATION — NOT TRAINING EVIDENCE**

## 1. Decision being learned

At each 100 ms decision epoch, the first agent selects exactly one of the 72
registered `SPLIT` profiles:

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

The initial observation contract contains normalized values plus an explicit
availability bit for every telemetry group:

- current and lagged PUSCH SNR, MCS, delivered throughput, and recent radio
  delivery outcomes;
- latest installed-frame lag, oldest-pending-frame lag, pending age/count,
  time since the latest useful installation, and availability flags;
- edge busy/pending state and the most recent explicit terminal outcome;
- previous action, payload, installation AoI, and action-switch indicator; and
- causal ego/radar risk summaries that are available before action selection.

The authored network-profile name, future SNR, ground truth, and the current
frame's tail output are forbidden policy inputs.

## 3. Auxiliary channel forecast

The supervisor's forecasting suggestion is implemented as multi-task learning,
not as access to future information. An auxiliary head reads the LSTM state and
predicts the next observed channel value or a small distribution over it:

$$
\widehat{s}_{t+1}=g_\psi(h_t).
$$

During training, the next *observed* SNR supplies a supervised target after the
transition occurs. A robust first loss is normalized Huber error:

$$
L_{\mathrm{forecast}}
=\operatorname{Huber}\!\left(
\widehat{s}_{t+1},s_{t+1}
\right).
$$

The combined optimization objective is

$$
L=L_{\mathrm{PPO}}+c_VL_V+c_FL_{\mathrm{forecast}}-c_H\mathcal{H}(\pi).
$$

The forecast head regularizes the recurrent representation and gives the policy
a learned short-horizon channel belief. Its predicted value may be exposed to
the policy head; the true future value may never be exposed. An ablation without
the forecast loss is required later to establish whether it helps.

## 4. Transition and feedback semantics

Each transmitted frame opens a pending ticket keyed by session, UE, frame and
action identity. If the next frame arrives first, the previous ticket remains
`PENDING`; it is not inferred to be lost. The policy may choose the next action
using pending age/count and installed-frame lag. Delayed or out-of-order
feedback closes the exact ticket it names.

Each ticket ultimately receives exactly one terminal outcome. The learning
environment keeps these classes separate:

- useful map installation;
- completed but non-newer installation;
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
LSTM state when the ticket opens. Generalized-advantage estimation may consume
only the terminally reconciled contiguous prefix, preserving on-policy credit
when rewards arrive out of order.

## 5. Initial latency-discounted quality reward

The first implementation does not feed an unbounded raw frame number to the
network or make frame ID an arbitrary scalar reward. Frame identity provides
credit attribution and derives the causal installed-frame lag

$$
\Delta f_t=f_t-f_{\mathrm{latest\ installed}}.
$$

For the action ticket belonging to frame $j$, the initial reward is

$$
r_j=
\mathbf{1}_{\mathrm{installed}}
\alpha Q_j\exp(-L_j/\tau)
-\beta_D\mathbf{1}_{\mathrm{transport\ failure}}
-\beta_B\frac{B_j}{B_{\max}}
-\beta_C\frac{C_j}{C_{\max}}
-\beta_S\mathbf{1}[a_j\ne a_{j-1}].
$$

$L_j$ ends at direct spatial-map installation; later controller-feedback
latency is not physical map AoI. `SUPERSEDED_PENDING` earns no fictitious
quality and no radio-loss penalty, but bytes and compute already spent remain
charged. This represents freshness through install latency in reward and frame
lag in the next causal observation without double-counting a second AoI term.

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
map-installation age.

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
4. evolves per-object map AoI and explicit terminal feedback;
5. withholds authored profile identity and future SNR from the policy;
6. separates training traces/seeds from evaluation traces/seeds; and
7. reproduces measured fixed-action cell statistics within preregistered
   tolerances before any PPO result is accepted.

Only after that reproduction gate passes should the LSTM-PPO policy and its
forecast-head ablation be trained.
