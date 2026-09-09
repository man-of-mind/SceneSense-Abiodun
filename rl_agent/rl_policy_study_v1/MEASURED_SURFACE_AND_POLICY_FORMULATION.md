# SplitFusion measured action surface and initial policy formulation

Status: **Steps 5–6 design complete; training not yet authorized**

This is chapter 2 of the Task-1 study. It consumes the completed 288-cell
campaign and its independently verified Steps 1–4 consolidation. It identifies
descriptive action relationships, then freezes the initial *shape* of the
split-only controller. It does not delete an action, train a policy, claim a
100-ms service, or invent `LOCAL`/`SKIP` outcomes.

The selected starting agent is:

> PPO with a GRU, an autoregressive 4×3×6 categorical SPLIT policy,
> technical-executability masks, and separate Lagrangian cost critics.

This sentence composes established methods. “Recurrent factorized masked PPO”
is convenient project shorthand, not the name of a new algorithm.

## 1. What changed after the 288 measurements

Before the campaign, we knew the payload and frozen model quality of every
action. We now also have live OAI/CARLA outcomes for every action under all four
network processes:

```text
72 split actions × 4 network profiles = 288 cells
```

The important campaign facts for policy design are:

- 896,856 frames were sent and 334,174 maps were installed;
- 66 cells had zero delivery;
- no frame installed within the 100-ms reference;
- 329,393 frames received feedback within the separate 500-ms timeout;
- median-of-cell-median install AoI was 364.5 ms;
- every published map was installed, so the main losses occurred earlier;
- one-way feature-uplink latency is not derivable from the registered clocks;
- the full UE-to-result round trip and compatible-clock residuals are
  available; and
- all 72 actions remain technically executable.

This rules out two tempting but incorrect designs:

1. A binary “within 100 ms” reward would be zero for every action and supply no
   learning signal.
2. Calling the derived round-trip residual “one-way uplink latency” would train
   on a quantity that was never measured.

The 100-ms number remains useful as a reporting reference. It is not currently
a feasible hard service constraint: the median frozen-tail computation alone
is about 111.6 ms, before sensor preparation, UE dispatch, queueing, radio,
serialization and map installation.

## 2. Step 5: three different meanings of dominance

An action `a` Pareto-dominates action `b` only if `a` is no worse in every
registered coordinate in the comparison and strictly better in at least one.
Three diagnostics answer three different questions.

### 2.1 Quality–payload diagnostic

Minimize measured payload while maximizing:

- preservation gates passed;
- absolute-service gates passed;
- localization gates passed; and
- segmentation installability.

The nondominated action IDs are:

```text
12, 44, 49, 50, 51, 66, 67, 69, 70, 71
```

This reproduces the earlier ten-action catalog diagnostic using the completed
live payload summaries. It is not a recommended reduced catalog.

### 2.2 Robust transport diagnostic

Across all four profiles, minimize payload and conditional install AoI while
maximizing reassembly, installation and feedback-within-500-ms rates. A
zero-delivery latency cell is treated as positive infinity, not as an
improbably fast blank value.

The robust transport front is:

```text
41, 46, 53, 59, 64, 65, 69, 70, 71
```

### 2.3 Joint diagnostic

Combining all registered quality capabilities and all four-profile transport
coordinates leaves 26 nondominated actions. That larger front is expected:
adding competing objectives makes exact dominance harder.

The intersection of the quality–payload and transport fronts is much more
informative:

| Action | Profile | Mean measured payload | Role |
|---:|---|---:|---|
| 69 | `AE32/UINT4/q=.70` | 79.0 KiB | sole non-emergency intersection action |
| 70 | `AE32/UINT4/q=.90` | 27.4 KiB | emergency/stress action |
| 71 | `AE32/UINT4/q=.98` | 6.1 KiB | extreme emergency/stress action |

Action 69 is therefore an important empirical knee, but it is not declared
universally optimal. It does not install segmentation, and its validation
quality is lower than larger actions.

The most relevant quality-side examples are:

| Action | Profile | Payload | Install/sent F/M/A/R | Interpretation |
|---:|---|---:|---|---|
| 12 | `noAE/UINT4/q=0` | 1,444.6 KiB | .007/.000/.000/.000 | highest registered gate counts, unusable transport in three profiles |
| 44 | `AE64/UINT6/q=.50` | 479.2 KiB | .570/.465/.000/.421 | segmentation-installable, but zero adverse delivery |
| 49 | `AE64/UINT4/q=.30` | 348.3 KiB | .617/.563/.235/.556 | strongest four-profile segmentation-installable compromise |
| 50 | `AE64/UINT4/q=.50` | 257.1 KiB | .605/.580/.490/.601 | localization-oriented compromise without segmentation installation |
| 51 | `AE64/UINT4/q=.70` | 158.1 KiB | .630/.612/.540/.601 | smaller non-emergency localization profile |
| 66 | `AE32/UINT4/q=0` | 230.0 KiB | .650/.593/.558/.628 | strong q=0 object profile, segmentation gate not fully met |
| 69 | `AE32/UINT4/q=.70` | 79.0 KiB | .660/.647/.611/.599 | robust non-emergency transport knee |
| 70 | `AE32/UINT4/q=.90` | 27.4 KiB | .635/.635/.573/.649 | emergency quality–delivery trade-off |
| 71 | `AE32/UINT4/q=.98` | 6.1 KiB | .661/.663/.626/.635 | smallest payload and weak emergency perception |

`F/M/A/R` means favorable, mid-variable, adverse and fade-recovery. These are
descriptive outcomes from one route realization per action/profile. There is
no replicate-based uncertainty test, so tiny differences are not scientific
proof that one action is universally superior.

## 3. Conditional feasibility is not technical invalidity

The evidence supports several labels, which must remain separate:

- **Technically executable:** all 72 actions.
- **Observed positive delivery in a particular profile:** 222/288 cells.
- **Observed zero delivery:** 66/288 cells.
- **Positive delivery in all four profiles:** 43 actions.
- **Positive delivery in all four and non-emergency:** 15 actions.
- **Observed installation within 100 ms:** zero actions/cells.
- **Segmentation installable:** a separate model capability.
- **Emergency/stress:** a quality/operating classification, not corruption.

Only the first item belongs in the invalid-action mask. Historical zero
delivery, Pareto dominance, high payload, emergency status and a predicted
deadline miss are learned or shielded performance outcomes—not reasons to
pretend the action cannot execute.

This is mathematically important. With technical mask `M_t(a)`, the policy is

$$
\pi_M(a\mid h_t)=
\frac{M_t(a)\exp z_a}{\sum_b M_t(b)\exp z_b}.
$$

In the current catalog, `M_t(a)=1` for all 72 actions unless a runtime bundle,
decoder, device, memory allocation or protocol-integrity condition is genuinely
unavailable. The Step-5 dominance labels never alter `M_t`.

## 4. Why the initial policy is split-only

The architecture chapter introduced a future top head with `SPLIT`, `LOCAL`
and `SKIP`. The completed campaign measures only SPLIT. Inventing transitions
for the other two modes would make the mathematics look complete while making
the evidence false.

The first implementation therefore chooses only one of the 72 measured split
actions. The future mode head remains an interface extension:

```text
v1 now:       SPLIT → family → quantizer → measured q
later: mode {SPLIT, LOCAL, SKIP}
                 └── SPLIT → family → quantizer → q
```

`LOCAL` enters only after CPU/GPU latency, energy, quality and compact-result
delivery are measured. `SKIP` enters only after its map-age transition,
new-object limitation and conservative guard are frozen. Continuous q remains
deferred until unseen q values validate interpolation within every categorical
branch.

## 5. Step 6: causal observation

The policy acts after aligned capture but before the common model front. The
lean v1 observation remains seven values:

1. `freshness_slack_s`;
2. `radar_risk`;
3. `ego_speed_mps`;
4. `ul_capacity_lcb_bps`;
5. `in_flight_age_s`;
6. `local_compute_slack_ms`; and
7. `time_since_last_processed_s`.

### 5.1 AoI is maintained per object

For object track `j`, map age is

$$
A_{j,t}=t-t^{capture}_{j,newest\ installed}.
$$

It is neither the age of the fastest object nor one frame-wide timestamp.
Each object can have a different newest contribution. A conservative error
model is

$$
E^{bound}_{j,t}=E^{model}_{j,a}+|v^{relative}_{j,t}|A_{j,t}.
$$

The policy does not receive every track as a variable-length input in v1.
Instead, an audited reducer turns the internal per-object values into the
worst relevant `freshness_slack_s`, approximately

$$
S_t=\min_j\left[
\frac{\epsilon_j-E^{model}_{j,current}}
     {\max(|v^{relative}_{j,t}|,v_{floor})}
-A_{j,t}
\right].
$$

Negative slack means the current map has freshness debt. The exact eligible
track set, risk weights, empty-map behavior, velocity floor and clipping still
require a small prospective freeze before materializing training data.

### 5.2 Why recurrence remains justified

An observed capacity estimate or SNR snapshot does not reveal whether the
channel is entering or leaving a fade. Recent delivery, in-flight age and
freshness trend also matter. A GRU produces

$$
h_t=f_\theta(h_{t-1},o_t),
$$

then all action and value heads use `h_t`. Hidden state resets at episode
boundaries and recurrent minibatches must preserve temporal order.

Network-profile name, target/future SNR, GT, route ID, absolute frame ID and
the current action's not-yet-produced model result remain forbidden inputs.

## 6. Factorized 72-action policy

The action is represented by three conditional categorical choices:

$$
\pi(a_t\mid h_t)=
\pi_f(f_t\mid h_t)
\pi_b(b_t\mid h_t,f_t)
\pi_q(q_t\mid h_t,f_t,b_t),
$$

where family has four values, quantizer has three and q has six. The joint log
probability used by PPO is

$$
\log\pi(a_t\mid h_t)=
\log\pi_f+\log\pi_b+\log\pi_q.
$$

Because the heads are conditional, this factorization can represent an
arbitrary distribution over the 72 combinations; it does not assume family,
quantizer and q are independent. It also exposes the engineering structure and
provides a clean future replacement of the discrete q head by a bounded
continuous head after dense-q validation.

The initial implementation uses a 64–64 MLP, a 128-unit GRU, family and
quantizer embeddings, three policy heads, one reward-value head and two cost
value heads. Those dimensions are modest starting values, not a scale claim.

## 7. Reward and constraints

The reward scores what the map contains after the outcome resolves. A selected
high-quality profile earns no quality credit if it does not install.

An initial objective is

$$
r_t=U_{map,t+1}
-\lambda_{air}\tilde C_{air,t}
-\lambda_{cmp}\tilde C_{compute,t}
-\lambda_{sw}\mathbf{1}[a_t\ne a_{t-1}].
$$

`U_map` gives primary value to usable vehicle/pedestrian coverage and
localization, then a smaller segmentation value only when that layer is
installable. A drop retains and ages the prior map.

Two explicit costs have separate critics:

$$
C_{coverage,t}=\frac{\sum_j w_j\mathbf{1}[j\text{ lacks a usable map match}]}
                         {\sum_j w_j},
$$

and

$$
C_{loc,t}=\frac{\sum_j w_j
  \left[E^{bound}_{j,t+1}/\epsilon_j-1\right]_+}
 {\sum_j w_j}.
$$

GT may compute these terms during offline training/evaluation, after the
action. GT never enters the observation. A Lagrangian PPO objective is

$$
J(\theta)-\eta_{cov}(J_{cov}-d_{cov})
-\eta_{loc}(J_{loc}-d_{loc}),\qquad \eta\ge0.
$$

This is constrained learning, not a runtime proof. Protocol integrity remains
deterministic.

### 7.1 Treatment of 100 ms

Do not use `1[AoI <= 100 ms]` as the main reward: the measured value is zero
everywhere. Retain it as a headline diagnostic and normalize continuous AoI or
freshness against 100 ms. Per-object age already affects localization debt, so
do not also add a full-strength generic delivery bonus and another full-strength
AoI penalty for the same event.

The 500-ms value remains an ACK accounting timeout. It is not a successful
service threshold.

### 7.2 Resource cost

Prefer measured PRB-seconds or airtime. The same bytes are more costly at low
spectral efficiency, so bytes alone are only a fallback proxy. Reward weights
and cost budgets are deliberately not numerically frozen until the training
split supplies normalization scales and a one-term-at-a-time sensitivity test
shows that no term dominates merely by units.

## 8. The 288 table is not directly a PPO replay buffer

Each campaign cell held one action fixed for a complete route. It did not log
behavior-policy probabilities or observe queue transitions caused by switching
actions. Therefore direct “offline PPO” on those rows would be scientifically
invalid.

The correct first training pipeline has two stages.

### 8.1 Supervised transition/emission model

Materialize one row per causal decision opportunity with:

- the seven pre-decision state values and their availability audit;
- the chosen action and exact technical mask;
- post-action reassembly, tail, install and ACK outcomes;
- measured payload, datagrams and resource use;
- compatible-clock durations and explicitly labelled residuals;
- the resulting map-age transition; and
- reward/cost terms computed after the action.

Fit an action-conditional stochastic model for quantities such as

$$
P(install\mid o_t,a_t),
$$

conditional latency distributions and resource cost. The unavailable one-way
uplink is not a target.

### 8.2 Trace-driven emulator plus PPO

The emulator advances causal network observations, bounded queues, in-flight
state and per-object map age. PPO may then generate switching trajectories
against this fitted environment.

Complete temporally correlated blocks—not random individual frames—form data
splits. The four profile labels may stratify evaluation but never enter the
policy. Final generalization requires independent trace seeds or later live
episodes; training and reporting on the identical four traces would measure
memorization.

## 9. What is implemented now

- `analyze_action_surface.py` reproduces the Step-5 fronts and writes explicit
  conditional labels without changing the action mask.
- `POLICY_CONTRACT_V1.json` freezes the initial split-only interface and lists
  every remaining evidence gate.
- `initial_split_policy.py` implements the GRU actor-critic and factorized
  categorical policy, including technical masks and reward/cost values.
- `test_initial_split_policy.py` contains five small hand-checkable CPU tests.

No PPO optimizer, transition surrogate or training run is included yet. That
is intentional: the causal seven-feature training table has not been
materialized and its three unresolved reducers must not be filled with
post-action or privileged data for convenience.

## 10. Immediate next gate

Before starting training:

1. audit whether every proposed state feature has a timestamp no later than
   the decision cutoff;
2. freeze `freshness_slack_s`, `radar_risk` and `local_compute_slack_ms` reducers;
3. prove that `ul_capacity_lcb_bps` is available through a UE-visible causal
   path rather than only the gNB collector;
4. materialize the per-frame transition table from the hash-bound evidence;
5. split by complete correlated blocks and reserve independent trace seeds;
6. fit and validate the stochastic outcome model; and
7. only then run PPO and compare it later with the required simple baselines.

On an RTX 5090, neural PPO optimization inside a lightweight validated emulator
will probably take hours rather than days. A realistic first estimate is 1–6
hours for one development run and roughly 12–48 GPU-hours for multiple seeds
plus reward sensitivity. The scientifically expensive part is validating the
emulator and held-out causal behavior, not the policy network's matrix
multiplications.

## 11. Supervisor summary

The project now has a complete measured action response surface. Compression
creates a real delivery/quality frontier: action 69 is the only non-emergency
action on both the registered quality–payload and robust transport fronts,
while actions 70–71 extend the frontier as emergency modes. This does not
justify discarding the other actions or claiming a 100-ms service. The initial
controller is an established recurrent constrained-PPO construction over all
72 measured split actions. Its novelty is the measured multimodal split/OAI/map
decision problem and causal freshness formulation—not a renamed PPO algorithm.
