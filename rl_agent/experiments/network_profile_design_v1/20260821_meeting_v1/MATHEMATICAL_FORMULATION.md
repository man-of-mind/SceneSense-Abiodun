# Gaussian and Markov network-profile formulation

## Slide-level formulation

At each 100-ms decision interval, the generator produces one **target SNR** value.

### 1. Shifted, bounded Gaussian

For a simple profile:

```text
gamma_k ~ TruncatedNormal(mu, sigma^2; L, U)
```

- `mu` shifts the bell curve left or right and controls the typical channel level.
- `sigma` controls how concentrated or variable the values are.
- This meeting design uses provisional illustration bounds `L=8.5 dB`, `U=24.5 dB`.

The normalized density is:

```text
f(gamma) = phi((gamma-mu)/sigma)
           / {sigma [Phi((U-mu)/sigma) - Phi((L-mu)/sigma)]},  L <= gamma <= U
```

where `phi` and `Phi` are the standard-normal PDF and CDF.

### 2. Markov state memory

Let `Z_k` be ADVERSE, INTERMEDIATE, or FAVORABLE:

```text
Pr(Z_(k+1)=j | Z_k=i) = P_ij
gamma_k | Z_k=i ~ TruncatedNormal(mu_i, sigma_i^2; L, U)
```

The diagonal `P_ii` is the probability of remaining in the same state. At `Delta t=0.1 s`:

```text
expected dwell time in state i = 0.1 / (1 - P_ii) seconds
```

The long-run state proportions satisfy:

```text
pi = pi P,     sum_i pi_i = 1
f_profile(gamma) = sum_i pi_i f_i(gamma)
```

Thus Gaussian parameters determine values *within* a state, while the Markov matrix determines how often and how long states occur.

Samples are conditionally independent within a state in this first design. Temporal memory comes only from the Markov state, avoiding an additional unvalidated smoothing parameter.

## Route-time convention

The qualified loop is `338.023 m` and `62.4 s`.
There are exactly 624 target values:

```text
I_k = [0.1 k, 0.1 (k+1)),   k = 0,...,623
```

The intervals cover `[0,62.4)` seconds. The final plotted boundary at 62.4 s extends the last held value; it is not sample 625.

## Four proposed profiles

| Profile | Stationary A/I/F | Expected dwell A/I/F | Expected switches / loop | Interpretation |
|---|---:|---:|---:|---|
| `FAVORABLE_STABLE` | 4%/16%/80% | 0.5s/0.7s/5.0s | 30 | 80% favorable occupancy; long favorable runs; rare brief fades |
| `MID_VARIABLE` | 20%/60%/20% | 0.7s/1.0s/0.7s | 75 | 20/60/20 occupancy; rapid switching among the three channel states |
| `ADVERSE_STABLE` | 80%/16%/4% | 5.0s/0.7s/0.5s | 30 | 80% adverse occupancy; long adverse runs; rare brief recovery |
| `FADE_RECOVERY` | 20%/60%/20% | 3.3s/5.0s/3.3s | 15 | Same 20/60/20 occupancy as mid-variable; five-times longer state dwell |

## Interpretation boundary

These figures are proposed, fixed-seed **target-SNR design traces**. They are not measured radio traces and the provisional bounds are not accepted operating limits. The same saved trace should be replayed for every compared action. Later replicates use new complete seeds, and a SCAN/Sionna-derived trace remains a held-out spatial test.
