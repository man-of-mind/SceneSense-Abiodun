# Presenter guide — network-profile figures

## The story in four slides

### Slide 1 — Gaussian mean shift

**Say:** A Gaussian profile gives us a controlled way to make some SNR values more common than others. Moving the mean toward the upper bound creates a favorable profile; moving it toward the lower bound creates an adverse one. The variance controls how wide or stable the profile is.

**Do not say:** that these illustrative bounds are already validated network limits.

### Slide 2 — Why add Markov memory?

**Say:** Independent Gaussian draws forget the previous 100-ms value. A Markov state adds persistence: the channel normally stays in its current condition and occasionally transitions. The diagonal transition probability directly determines expected dwell time.

### Slide 3 — Four proposed profile distributions

**Say:** The profiles are not four fixed SNR values. Favorable and adverse change long-run state occupancy. Mid-variable and fade/recovery deliberately have the same long-run SNR distribution, but fade/recovery scales transition rates down by five. This isolates rapid variation from sustained fades.

### Slide 4 — One route-loop trace per profile

**Say:** The qualified `338.0-m` loop takes `62.4 s`, giving exactly 624 decisions at 100-ms cadence. Each colored step is one target held for a single decision interval. We save each trace once and replay the exact same values for all action profiles, preserving a fair comparison.

## One-sentence recommendation

Use the Markov-modulated Gaussian model for the main experiment because it retains the supervisor's intuitive mean/variance control while adding the temporal persistence needed to study queue build-up, fades, and recovery; keep an IID bounded-Gaussian trace as a diagnostic control.

## Figure mapping

- `01_gaussian_mean_shift`: explains mean and variance.
- `02_markov_model`: explains temporal memory and dwell time.
- `03_markov_transition_matrices`: technical backup for the four matrices.
- `04_profile_marginal_distributions`: compares long-run profile shapes.
- `05_target_snr_trace_overview`: compares all four 62.4-s traces.
- `profiles/*_card`: one distribution plus its route trace for a dedicated slide.

Every figure is available as PNG for convenient insertion and PDF/SVG for vector-quality editing.
