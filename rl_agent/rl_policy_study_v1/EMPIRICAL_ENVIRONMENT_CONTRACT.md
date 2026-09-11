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
- current map AoI summaries, critical-track AoI, time since the latest useful
  installation, and freshness slack;
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

Each transmitted frame receives exactly one terminal outcome. The learning
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

## 5. Map-utility reward

Let $\mathcal{O}_t$ be the tracked objects in the map after the transition. For
object $i$, let $q_{i,t}\in[0,1]$ be an auditable task-quality score,
$A_{i,t}$ its AoI, $w_i$ its criticality weight, and $\tau_i$ the registered
freshness time constant. Define map utility

$$
U_t=
\frac{\sum_{i\in\mathcal{O}_t}
w_i q_{i,t}\exp(-A_{i,t}/\tau_i)}
{\sum_{i\in\mathcal{O}_t}w_i+\varepsilon}.
$$

The immediate reward uses the *change* in post-outcome map utility:

$$
r_t =
(U_{t+1}-U_t)
-\lambda_B\frac{B_t}{B_{\max}}
-\lambda_C\frac{C_t}{C_{\max}}
-\lambda_S\mathbf{1}[a_t\ne a_{t-1}].
$$

- $B_t$ is the feature traffic charged to this action, including bytes spent on
  a later-superseded frame.
- $B_{\max}$ is a frozen normalization constant, initially the largest measured
  registered action payload—not a bandwidth limit.
- $C_t$ and $C_{\max}$ similarly normalize measured compute expenditure.
- The switching term discourages oscillation without banning useful changes.

The exponential factor equals one for a fresh object and decays smoothly as it
ages. At $A_{i,t}=\tau_i$, freshness contributes $e^{-1}\approx0.368$ of its
fresh value. This avoids turning the borrowed 100 ms service reference into an
unsupported universal safety cliff.

Reward weights and $\tau_i$ values are not frozen here. They require a
normalization and sensitivity study on training-only transitions. Safety or
resource requirements that must hold are represented as separately reported
cost signals rather than hidden inside a single reward number.

## 6. Evidence that can and cannot train this environment

The completed 288-cell live campaign is authoritative for the original runtime
and provides all 72 actions under all four network processes. The optimized
latest-only 288-cell counterfactual preserves measured per-cell reassembly and
admission totals and recomputes map outcomes after the edge optimization.

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
