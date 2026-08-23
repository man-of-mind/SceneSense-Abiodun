# Presenter guide — network-profile figures

> Label every value as target-SNR design data. These are not measured achieved-OAI traces, and target-to-RFsim actuation remains a separate calibrated mapping.

## The story in four slides

### Slide 1 — Gaussian mean shift

**Say:** The qualified design band is `5.5` to `24.5 dB`. The unchanged relative rule places the adverse, intermediate, and favorable means at `8.35`, `15.00`, and `21.65 dB`, with sigma `1.14 dB`.

### Slide 2 — Why add Markov memory?

**Say:** Independent Gaussian draws forget the previous 100-ms value. A Markov state adds persistence: the channel normally stays in its current condition and occasionally transitions. The diagonal transition probability directly determines expected dwell time.

### Slide 3 — Four preserved profile distributions

**Say:** The profiles are not four fixed SNR values. Favorable and adverse change long-run state occupancy. Mid-variable and fade/recovery deliberately have the same long-run SNR distribution, but fade/recovery scales transition rates down by five. This isolates rapid variation from sustained fades.

### Slide 4 — Route B reference traces

**Say:** Route B is `1268.68 m`. The longest qualified density duration is `419.90 s`, so the presentation uses a `420 s`, `4200`-sample reference trace at 100-ms cadence.

**Say:** For fair action-profile comparison, each network profile has one trace ID and fixed seed. Every one of the 72 action profiles and every density restarts that sequence at sample zero. Low and medium stop on their shorter Route B completion; dense uses approximately the full reference. If a run exceeds 420 seconds, the stateful generator continues deterministically until episode end.

**Do not say:** that RFsim accepts target PUSCH SNR directly. A calibrated target-to-RFsim mapping is a separate actuation layer.

## Figure mapping

- `01_gaussian_mean_shift`: explains mean and variance.
- `02_markov_model`: explains temporal memory and dwell time.
- `03_markov_transition_matrices`: technical backup for the four matrices.
- `04_profile_marginal_distributions`: compares long-run profile shapes.
- `05_target_snr_trace_overview`: compares all four 420-s reference traces.
- `profiles/*_card`: one distribution plus its route trace for a dedicated slide.

Every figure is available as PNG for convenient insertion and PDF/SVG for vector-quality editing.
