# Track A experiment index

Experiment directories are immutable siblings. Canonical runs for the current implementation:

- `deterministic_acceptance/20260810_182018` — current four-scenario acceptance run.
- `pilot/20260810_182039` — current corrected real-replay pilot and source of `../../POLICY_RESULTS.md`.

Earlier diagnostic runs are intentionally retained for provenance:

- `deterministic_acceptance/20260810_180355` — stopped before results when the original raw reward-gap
  assertion was found invalid for lexicographic safety; contains no claimed result.
- `deterministic_acceptance/20260810_180443` — superseded after separating safe-only reward regret from
  safety/action-set mismatch.
- `pilot/20260810_180620` — superseded diagnostic that exposed a replay validity-gate mismatch. It scored
  off-FOV actors and used bbox-center GT; the corrected run uses the validated in-frustum, origin-coordinate,
  score>=0.20, greedy 5 m association convention.
- `deterministic_acceptance/20260810_181011` and `pilot/20260810_181029` — superseded after adding the
  actual capture-wait interval to the safety-bound evaluation.
- `deterministic_acceptance/20260810_181317` and `pilot/20260810_181333` — superseded after adding explicit
  C1 estimate-miss reporting and changing the figure to the tracked-object C2 population.
- `deterministic_acceptance/20260810_181557` and `pilot/20260810_181615` — superseded after fixing the exact
  two-step telemetry-lag index and preventing hidden GT objects from seeding delivered map contributions.

Do not combine or average superseded and canonical runs.
