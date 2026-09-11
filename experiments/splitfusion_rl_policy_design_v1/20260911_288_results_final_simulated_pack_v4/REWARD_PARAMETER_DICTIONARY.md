# Reward formulation — symbol dictionary

The initial split-only policy uses installed-map utility change plus explicit
resource costs:

$$
r_t = (U_{t+1}-U_t)
      - \lambda_B\frac{B_t}{B_{\max}}
      - \lambda_C\frac{C_t}{C_{\max}}
      - \lambda_S\mathbf{1}[a_t \ne a_{t-1}],
$$

with

$$
U_t =
\frac{\sum_{i\in\mathcal O_t} w_i Q_{i,t}
      \exp\!\left(-A_{i,t}/\tau_i\right)}
     {\sum_{i\in\mathcal O_t} w_i + \varepsilon}.
$$

| Symbol | Meaning |
|---|---|
| $t$ | Decision epoch. |
| $a_t$ | One selected split action among the 72 registered profiles. |
| $r_t$ | Immediate reward assigned to the transition from epoch $t$ to $t+1$. |
| $U_t$ | Utility of the currently installed spatial map before the new transition. |
| $U_{t+1}-U_t$ | Actual improvement or degradation in installed-map utility; an undelivered update does not create fictitious quality credit. |
| $\mathcal O_t$ | Objects currently represented in the map utility calculation. |
| $i$ | One tracked map object. |
| $w_i$ | Application importance of object $i$, for example a larger weight for a vulnerable road user on the ego path. |
| $Q_{i,t}$ | Normalized quality/utility contribution of the installed observation for object $i$. This is distinct from the action's compression knob $q$. |
| $A_{i,t}$ | Age of information of object $i$: current time minus capture time of its newest installed observation. |
| $\tau_i$ | Freshness tolerance for object $i$; utility falls to $e^{-1}\approx0.368$ when $A_{i,t}=\tau_i$. |
| $\varepsilon$ | Small positive constant preventing division by zero when the map contains no weighted objects. |
| $B_t$ | Feature bytes actually charged to action $a_t$. |
| $B_{\max}$ | Fixed payload normalizer: the largest registered median action payload, not instantaneous channel capacity. |
| $C_t$ | Compute consumed by the action, including spent work on an intentionally superseded frame. |
| $C_{\max}$ | Fixed compute-cost normalizer. |
| $\lambda_B$ | Weight on communication cost. |
| $\lambda_C$ | Weight on compute cost. |
| $\lambda_S$ | Penalty for switching actions too frequently. |
| $\mathbf{1}[a_t\ne a_{t-1}]$ | Indicator equal to 1 when the action changes and 0 otherwise. |

The training return remains

$$
G_t = \sum_{k=0}^{\infty}\gamma^k r_{t+k},
$$

where $\gamma\in[0,1)$ controls how much future reward matters. The auxiliary
next-SNR forecast improves the recurrent representation; it does not replace
the PPO reward and never supplies future SNR to the acting policy.
