# Feature, history and output glossary

This is the actual deployed 22-feature feed-forward conditional hybrid actor. “History” is an explicit previous-decision/outcome input block, not an LSTM state. Source contracts are the retained `run4_contract.py` plus Run-5 SNR extension; hashes are in the results.

| Group | Zero-based indices | Input meaning / encoding |
|---|---|---|
| Scene | 0 | Camera spatial information (SI), centred/scaled by frozen empirical parameters. |
| Scene | 1 | Recorded radar P40 scene descriptor, passed directly as `radar_p40`. No new scene metric is derived. |
| Channel | 2 | Latest strictly prior UE-decoded round-0 UL MCS, divided by 28. Not measured PUSCH SNR. |
| Channel | 21 | Commanded causal SNR proxy, `(dB − 5.5) / 19`. GOOD=23.5 dB; BAD=8.5 dB. Not achieved/measured SNR. |
| Backlog | 3 | `log1p(pre-action RLC backlog bytes) / frozen backlog scale`. Timeline shows raw kB (1,000 bytes), not this encoded value. |
| Previous decision/outcome | 4–15 | Twelve-way one-hot of previous joint mode 0–11; all zero when absent. |
| Previous decision/outcome | 16 | Previous `q_e4 / 9800`. Distinct from plotted/output q, which is `q_e4 / 10000`. |
| Previous decision/outcome | 17 | Previous Q_perc if successful/defined; otherwise unavailable placeholder zero. A successful Q_perc can legitimately equal zero. |
| Previous decision/outcome | 18 | Previous action-open-to-feedback latency in ms / 170 if successful; otherwise placeholder zero. Not prediction-completion latency or STALE receipt timing. |
| Previous decision/outcome | 19 | `prev_present`: previous outcome/action record exists. |
| Previous decision/outcome | 20 | `prev_success`: retained successful/defined feedback, not merely a completed prediction or map publication. |

## History states used in the figures

| Display label | Present / success | Mode and q | Quality and feedback latency | Interpretation |
|---|---|---|---|---|
| Initial | 0 / 0 | Zero | Zero | First recorded decision in each run. |
| Cleared | 0 / 0 | Zero | Zero | Later empty history; in these captures each follows a recorded timely-undefined outcome. |
| Timeout | 1 / 0 | Retained | Unavailable zeros | Timeout/failure encoding; not measured zero quality/latency. |
| Defined | 1 / 1 | Retained | Valid values | Timely defined outcome, including possible defined zero quality. |

Counts over all recorded decisions, not only explanation targets:

| Run | Initial | Cleared after timely undefined | Timeout | Defined | Total |
|---|---:|---:|---:|---:|---:|
| V11 | 1 | 87 | 225 | 1458 | 1771 |
| V12 | 1 | 68 | 907 | 574 | 1550 |
| V13 | 1 | 54 | 685 | 793 | 1533 |
| V14 | 1 | 49 | 565 | 914 | 1529 |

Cleared counts are decision states, not necessarily distinct reset episodes. Their recorded event chain is used, not inferred from SNR. The retained engine breaks the session after excluded evaluator outcomes; the state builder then has no previous record. Returning history may restore action/presence without restoring quality. The 170-ms value is the original feedback deadline, not an inference-latency boundary or map-publication deadline.

## Outputs and attribution units

- **Mode:** categorical argmax among 12 joint-mode logits. Mode 11 is AE32/UINT4. A larger numeric mode ID is not more compression and is never regressed here.
- **Mode-m preference:** `logit_m − max(logit_other_modes)`, unitless logit-margin units; positive means m wins. For selected non-11 modes, computed equivalently as `log(p_m / max(p_other))` from retained full-precision probabilities.
- **Conditional qm:** deterministic mode-m head, mapped to its registered q bounds and quantized to wire precision, `q_e4/10000`. The decomposition includes this quantization. Conditional q11 is executed only when mode 11 is chosen. In these actions, larger q means greater spatial dropping; it is not payload bytes.
- **Observed payload:** timeline `payload_l2_bytes`, verified retained SFD4 payload count; kB=1,000 bytes. It is context, never a counterfactual actor output.
- **Timeline time:** retained `t_since_first_capture_s`, using the first decision frame's capture as origin. It locates decision frames, not actor execution duration or a latency measurement.
- **Exact grouped Shapley:** all 16 intact-group coalitions; average marginal contributions over all 24 group orders. Each output satisfies reference + four signed contributions = target, up to floating-point rounding. Groups share interactions; contribution to history does not isolate its individual fields.
- **Question A:** eight-reference average of signed group contributions, relative to one common balanced empirical background. Summary bars average the absolute values of those per-target averages, so donor-specific opposing effects can cancel before the absolute value.
- **Question B:** signed change from the immediately preceding recorded state, not the preceding plotted/sampled point. First-state targets have no predecessor and no B attribution.
- **Far:** a coalition outside the retained empirical full-state proximity radius. This is a warning, not proof that a state is impossible; “near” is not proof it occurred in training or could be generated dynamically. All coalitions remain included.

Machine-readable feature vectors, source/head identities, reference IDs and support flags are in `selection_01` and `attribution_01`. The per-decision CSV has one row per target/question, cohort flags and four outputs × four groups; the JSON adds exact pair/reference IDs. Coalition group order is channel, scene, backlog, history; visual order is scene, channel, backlog, history, always mapped by name.
