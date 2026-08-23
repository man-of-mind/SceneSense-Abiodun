# Gaussian and Markov network-profile formulation

> All values here are **target-SNR design values**, not measured achieved-OAI SNR. RFsim actuation requires a separate calibrated target-to-RFsim mapping; this design does not assume that RFsim accepts target PUSCH SNR directly.

## Bounded Gaussian emission

At each `100 ms` interval, the generator produces one target-SNR value. The qualified design band is `L=5.5 dB`, `U=24.5 dB`, with `R=19 dB`.

```text
gamma_k | Z_k=i ~ TruncatedNormal(mu_i, sigma_i^2; L, U)
```

The preserved relative rule gives state means A/I/F = `8.35, 15.00, 21.65 dB` and state sigmas A/I/F = `1.14, 1.14, 1.14 dB`.

The normalized density is:

```text
f(gamma) = phi((gamma-mu)/sigma)
           / {sigma [Phi((U-mu)/sigma) - Phi((L-mu)/sigma)]},  L <= gamma <= U
```

The summary computes each component's exact mean and variance after truncation, then combines them using the stationary state probabilities. `phi` and `Phi` are the standard-normal PDF and CDF.

## Markov state memory

Let `Z_k` be ADVERSE, INTERMEDIATE, or FAVORABLE:

```text
Pr(Z_(k+1)=j | Z_k=i) = P_ij
```

The transition matrices are unchanged. At `Delta t=0.1 s`, expected state dwell is:

```text
E[D_i] = Delta t / (1 - P_ii)
```

Stationary state occupancy and the profile marginal satisfy:

```text
pi = pi P,     sum_i pi_i = 1
f_profile(gamma) = sum_i pi_i f_i(gamma)
```

## Route B reference trace and runtime continuation

Route `Town10HD_Opt Route B full-map loop v1` is `1268.68 m`. Qualified durations are low `359.25 s`, medium `358.70 s`, and dense `419.90 s`.

The presentation artifact is a `420 s` reference prefix containing `4200` target values:

```text
I_k = [0.1 k, 0.1 (k+1)),   k = 0,...,4199
```

The plotted intervals cover `[0,420)` seconds. The `420 s` boundary extends the final held value and is not an additional sample.

At episode start, instantiate the profile generator with its fixed seed and replay from sample zero. Low and medium consume their matching prefix; dense consumes approximately the full reference prefix. If any episode lasts longer than 420 seconds, the same RNG and Markov state continue deterministically beyond sample 4199 until episode end. The reference length is not a runtime cap.

The same profile trace ID and seed are reused across all 72 action profiles and every density. A new random trace must not be generated per action-profile episode.

## Preserved profiles

| Profile | Trace ID | Seed | Stationary A/I/F | Expected dwell A/I/F | Expected switches / 420-s reference | Interpretation |
|---|---|---:|---:|---:|---:|---|
| `FAVORABLE_STABLE` | `GM_V2_FAVORABLE_STABLE_SEED_2026082101` | 2026082101 | 4%/16%/80% | 0.5s/0.7s/5.0s | 202 | 80% favorable occupancy; long favorable runs; rare brief fades |
| `MID_VARIABLE` | `GM_V2_MID_VARIABLE_SEED_2026082102` | 2026082102 | 20%/60%/20% | 0.7s/1.0s/0.7s | 504 | 20/60/20 occupancy; rapid switching among the three channel states |
| `ADVERSE_STABLE` | `GM_V2_ADVERSE_STABLE_SEED_2026082103` | 2026082103 | 80%/16%/4% | 5.0s/0.7s/0.5s | 202 | 80% adverse occupancy; long adverse runs; rare brief recovery |
| `FADE_RECOVERY` | `GM_V2_FADE_RECOVERY_SEED_2026082104` | 2026082104 | 20%/60%/20% | 3.3s/5.0s/3.3s | 101 | Same 20/60/20 occupancy as mid-variable; five-times longer state dwell |

## Interpretation boundary

These are fixed-seed **target-SNR design traces**. They are not measured achieved-OAI traces. Turning a target into RFsim controls remains a separate calibrated mapping step.
