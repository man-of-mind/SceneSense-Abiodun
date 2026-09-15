# Split-only recurrent PPO: preliminary architecture v1

Status: **DESIGN + NETWORK SKELETON ONLY — NO TRAINING CLAIM**

## Decision boundary

At every decision epoch the UE chooses one of the 72 measured split profiles:

$$
a_t \in \{0,\ldots,71\}.
$$

Each action fixes the feature family, quantizer and registered drop fraction
$q$. `LOCAL` and `SKIP` are deliberately excluded until their transitions are
measured. Poor but executable split actions remain learnable outcomes; the
action mask is reserved for actions that cannot physically execute.

## Why recurrence is needed

The current SNR is insufficient by itself. OAI channels evolve over time, and
the UE does not directly observe the next channel state or the complete edge
state. The LSTM summarizes causal history:

$$
(h_t,c_t)=\operatorname{LSTM}(e(o_t),h_{t-1},c_{t-1}).
$$

An auxiliary head predicts the next observed SNR from $h_t$. The true next SNR
becomes a training target only after the transition; it is never supplied to
the policy before the action.

In plain language, the LSTM gives the agent a short-term memory. At decision
epoch $t$, it receives the encoded current observation and its two memory
vectors from epoch $t-1$:

- the **cell state** $c_{t-1}$ carries longer-lived context, such as whether
  the channel has been steadily degrading or recovering;
- the **hidden state** $h_{t-1}$ is the compact summary exposed to the policy,
  value and forecast heads; and
- learned input, forget and output gates decide what new evidence to store,
  what old evidence to retain and what part of the memory to expose.

This matters because two epochs can have the same instantaneous SNR but imply
different decisions. A falling SNR sequence may favor a smaller payload before
congestion develops, whereas the same SNR during recovery may support a less
aggressive action. Recent delivery, supersession and map-AoI history also help
the memory distinguish a brief radio dip from a persistent service backlog.

## Causal observation groups

The first simulator should expose normalized values and availability flags for:

- current and lagged PUSCH SNR, MCS and delivered throughput;
- recent complete-message delivery and terminal-outcome history;
- latest installed-frame lag, oldest-pending-frame lag, pending age/count and
  time since a useful installation;
- edge busy/pending state plus recent supersession and expiry outcomes;
- previous action, payload, installation AoI and action-switch indicator; and
- causal ego/object-risk summaries available before action selection.

Forbidden inputs are the authored network-profile name, future SNR, CARLA
ground truth, and the current frame's eventual tail result.

The exact asynchronous attribution and missing-feedback behavior is frozen in
[AGENT_TIMELINE_AND_DELAYED_FEEDBACK_V1.md](AGENT_TIMELINE_AND_DELAYED_FEEDBACK_V1.md).
In particular, absence of feedback at the next frame means `PENDING`, not
`LOST`; later feedback closes the exact frame/action ticket.

## Network

`model.py` implements the initial network:

![Split-only recurrent PPO architecture](AGENT_ARCHITECTURE_V1.svg)

```mermaid
flowchart LR
    OBS["Causal observation<br/>SNR, MCS, throughput<br/>map AoI and risk<br/>delivery and prior action"]
    ENC["LayerNorm + MLP<br/>observation encoder"]
    LSTM["LSTM memory<br/>hidden state h(t)<br/>cell state c(t)"]
    POLICY["Policy head<br/>72 split-action logits"]
    VALUE["Reward-value critic<br/>expected return"]
    COST["Cost critics<br/>bytes, compute, switching"]
    FORECAST["Auxiliary forecast head<br/>next observed SNR"]
    ACTION["Executable split action<br/>family + quantizer + q"]
    ENV["Environment feedback<br/>map utility and AoI<br/>terminal outcome and costs"]

    OBS --> ENC --> LSTM
    LSTM --> POLICY -->|masked sample| ACTION --> ENV
    LSTM --> VALUE
    LSTM --> COST
    LSTM --> FORECAST
    ENV -->|next causal observation and reward| OBS

    classDef observation fill:#eaf1ff,stroke:#3366cc,stroke-width:2px;
    classDef memory fill:#fdebf0,stroke:#ef5675,stroke-width:2px;
    classDef head fill:#ecf8f5,stroke:#2a9d8f,stroke-width:2px;
    classDef environment fill:#f8eeee,stroke:#8c564b,stroke-width:2px;
    class OBS,ENC observation;
    class LSTM memory;
    class POLICY,VALUE,COST,FORECAST,ACTION head;
    class ENV environment;
```

This is PPO with recurrent state, hard executability masking and an auxiliary
forecast objective. It is a composition of established techniques, not a new
algorithm called “Recurrent Hierarchical Masked PPO.”

### How the LSTM plugs into PPO

The LSTM is not a separate controller placed in front of PPO. It replaces the
memoryless feature layer inside the actor-critic network:

1. The UE forms only the information available before choosing $a_t$.
2. The encoder converts that observation to a compact feature vector.
3. The LSTM updates $(h_t,c_t)$ from that feature and the previous memory.
4. The policy head converts $h_t$ into probabilities over the 72 split actions.
5. The value and cost heads use the same $h_t$ to estimate future reward and
   resource consequences for PPO training.
6. The forecast head predicts the SNR that will be observed at $t+1$. Its
   error is an auxiliary training loss; its prediction is not manually inserted
   into the observation and does not replace the policy output.

During training, rollouts retain the observation, chosen action, reward,
terminal flag and incoming LSTM state. PPO optimizes short contiguous
sequences so gradients can teach the memory which past signals matter. The
memory resets at the start of a new episode or UE session; in a future
multi-UE deployment, each UE keeps its own state so histories cannot leak
between vehicles.

The forecast loss is useful only as a representation aid. It encourages
$h_t$ to encode channel trend and temporal structure, which can improve action
selection under fading and recovery. The first ablation must compare the same
PPO architecture with and without this head; if it does not improve policy
return or robustness, it should be removed.

## Reward and intentional supersession

The equation below is the richer object-level utility candidate. The revised
first-training discussion draft is
[REWARD_FORMULATION_V2.md](REWARD_FORMULATION_V2.md); it keeps segmentation,
localization and latency separately weighted and uses frame ID for exact
causal attribution and frame-lag state rather than as an unbounded scalar
reward input. The asynchronous timing contract remains in
[AGENT_TIMELINE_AND_DELAYED_FEEDBACK_V1.md](AGENT_TIMELINE_AND_DELAYED_FEEDBACK_V1.md).

The reward should use the change in installed-map utility:

$$
r_t=(U_{t+1}-U_t)
 -\lambda_B\frac{B_t}{B_{\max}}
 -\lambda_C\frac{C_t}{C_{\max}}
 -\lambda_S\mathbf{1}[a_t\ne a_{t-1}].
$$

For tracked object $i$, a useful starting map utility is

$$
U_t=
\frac{\sum_i w_i Q_{i,t}\exp(-A_{i,t}/\tau_i)}
     {\sum_i w_i+\varepsilon}.
$$

Here $Q_{i,t}$ means installed-map quality for object $i$; it is deliberately
capitalized so it cannot be confused with the action's dropped-feature
fraction $q$.

### Reward-symbol dictionary

| Symbol | Meaning |
|---|---|
| $t$ | Current decision epoch. |
| $a_t$ | Selected split action: one of the 72 registered family/quantizer/$q$ profiles. |
| $r_t$ | Immediate transition reward. |
| $U_t$ | Utility of the spatial map before the transition. |
| $U_{t+1}-U_t$ | Measured improvement or degradation in installed-map utility. An update that is not installed creates no fictitious quality credit. |
| $i$ | One tracked object in the map. |
| $w_i$ | Application importance of object $i$, such as a larger weight for a vulnerable road user on the ego path. |
| $Q_{i,t}$ | Normalized installed quality/utility contribution for object $i$; this is not compression $q$. |
| $A_{i,t}$ | Object-level Age of Information: current time minus capture time of the newest installed observation for object $i$. |
| $\tau_i$ | Freshness tolerance. The freshness factor is $e^{-1}\approx0.368$ when $A_{i,t}=\tau_i$. |
| $\varepsilon$ | Small positive denominator guard when there are no weighted map objects. |
| $B_t$ | Feature bytes actually charged to the selected action. |
| $B_{\max}$ | Fixed payload normalizer—the largest registered median action payload, not instantaneous network capacity. |
| $C_t$ | Compute actually consumed, including spent work on an intentionally superseded frame. |
| $C_{\max}$ | Fixed compute-cost normalizer. |
| $\lambda_B$ | Communication-cost weight. |
| $\lambda_C$ | Compute-cost weight. |
| $\lambda_S$ | Action-switching penalty weight. |
| $\mathbf{1}[a_t\ne a_{t-1}]$ | Indicator equal to 1 when the action changes and 0 otherwise. |

PPO optimizes the discounted return

$$
G_t=\sum_{k=0}^{\infty}\gamma^k r_{t+k},
$$

where $\gamma\in[0,1)$ controls how strongly later consequences influence the
current decision.

A `SUPERSEDED_PENDING` frame receives no new-map utility because it was never
installed, but it is still charged for bytes and compute already consumed. It
must not be mislabeled as radio loss or structural failure. This lets the
policy learn that a large action can be wasteful when newer information is
likely to overtake it, without punishing the network for an intentional
latest-only scheduling decision.

## How the measurements enter the simulator

- The 288 cells provide action/profile-conditioned payload, delivery and
  original-runtime freshness evidence.
- The final v3 live comparison supplies prospective edge-service distributions
  for actions 30, 50 and 71. It must not be converted into one constant
  subtraction from all historical AoI values.
- The simulator must replay the two-stage latest-only scheduler so faster edge
  service changes supersession, completion and installation nonlinearly.
- Missing v3 actions require a conservative calibrated service model with an
  uncertainty flag, followed by fixed-action reproduction tests before PPO.
- Action 15's v3 non-finite live output is a fail-closed reliability outcome;
  it is not a latency sample and should be masked only if the corresponding
  executable path remains invalid at deployment time.

## Next implementation gate

Before PPO training, implement a causal trace-driven environment and require it
to reproduce held fixed-action statistics for payload, complete reassembly,
supersession, edge completion, installation probability and map AoI. Training
starts only after those checks pass. The first ablation compares the same PPO
network with and without the auxiliary next-SNR forecast loss.
