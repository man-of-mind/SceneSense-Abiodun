# SplitFusion Steps 5–6 measured policy-design evidence

This create-only bundle consumes the independently verified 72-action summary
with SHA-256
`250eb6d9391c0de5a343d692484647bbdb181153ba64f7f4a447b0076f580a88`.
It performs descriptive dominance and conditional-outcome classification only.
It does not alter the 72-action registry or authorize training/deployment.

## Step 5 results

- Actions: 72; conditional action/profile cells: 288.
- Technical policy mask: 72/72 allowed.
- Quality–payload Pareto IDs:
  `12,44,49,50,51,66,67,69,70,71`.
- Robust transport Pareto IDs:
  `41,46,53,59,64,65,69,70,71`.
- Intersection: `69,70,71`.
- Action 69 is the only non-emergency intersection member.
- Joint quality/transport Pareto actions: 26.
- Positive delivery in all four profiles: 43 actions, of which 15 are
  non-emergency.
- Zero-delivery cells: 66/288.
- Cells with an observed install inside 100 ms: 0/288.

Exact Pareto comparisons use one realized route per action/profile and have no
replicate-based uncertainty test. They are diagnostic labels, not pruning or
hard feasibility claims.

## Step 6 decision

The initial controller is split-only PPO with a GRU, an autoregressive
family/quantizer/q categorical action distribution, technical-executability
masks, one reward critic and two Lagrangian cost critics. `LOCAL`, `SKIP` and
continuous q are deliberately deferred because the completed campaign did not
measure their transitions.

The 288 aggregate table is not treated as a PPO replay buffer. Each route held
one action fixed, behavior probabilities are unavailable, and profile switching
was not observed. The next data step is a causal per-decision table followed by
an action-conditional stochastic emulator. The unavailable one-way feature
uplink is never synthesized as a training target.

## Artifact bindings

| Artifact | Rows | SHA-256 |
|---|---:|---|
| `action_dominance_diagnostics.csv` | 72 | `28857ff039a15ea8b2ec99ee5ad530015498c269e6776d59c7d2b5cb5688fd61` |
| `conditional_outcome_labels.csv` | 288 | `599f5a22a83b943d6e2e32ce0ddc7dfcfaa1d8b58a429249ac8b0947324e1677` |
| `step5_summary.json` | — | `d56d0f3f2bcf6b02cf78c0ace2964671477fb9c5fcd5cb05501e333088748c66` |
| `MEASURED_SURFACE_AND_POLICY_FORMULATION.md` | — | `0b14ffe75c0ad39919a83431e33639b293b3d731e0284d7094f59eab962f7738` |
| `POLICY_CONTRACT_V1.json` | — | `1d608d7220cb11a1abd1773954fb5c7162cdae5b67236e79b35578b617a17ce9` |
| `initial_split_policy.py` | — | `d84d738a20c779885611baee867ffdf6fbbe4efe6b852bb0249adcdfd0830e72` |

The complete scientific interpretation, mathematics and unresolved gates are
in `rl_agent/rl_policy_study_v1/MEASURED_SURFACE_AND_POLICY_FORMULATION.md`.

No CARLA, OAI, Docker, RFsim, CUDA, inference, dataset access or training was
performed.
