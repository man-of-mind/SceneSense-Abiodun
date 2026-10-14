# Frozen-agent behaviour across M3.1–M3.4

Completed offline, CPU-only. The evidence is mixed, not a single channel-adaptation story: **scene leads the common-background contrasts; history leads average sampled step changes; individual changes can instead be led by scene or backlog.** These answer different questions, and neither establishes live causality or optimality.

The deployed actor is the 22-input feed-forward conditional hybrid actor, not an LSTM. Weights SHA256: `bb9b0d256de79b4498cd622e1daa59edcf43ff3072dc9f675b265aac579ed3ce`. V13's original registered **STOP remains unchanged**. The failed uniform q+1000 quality screen stays closed.

## 1. Constant GOOD/BAD still permits changing decisions

Against the same eight-state GOOD/BAD-balanced background (question A), scene has the largest mean absolute contribution to both mode-11 margin and conditional q in all four runs. Channel is nonzero, including constant-SNR runs: these references cross channel conditions. Against preceding decisions (question B), history leads the sampled average in every run, with scene and intermittent backlog also contributing.

Each cell below is `mode-11 logit-margin units / q fraction`, question A, 100 uniformly spaced targets per run. Magnitudes are averaged **after** averaging signed contributions over references; not percentages of causation.

| Full Town10 map condition | Scene | Channel | Backlog | History |
|---|---:|---:|---:|---:|
| M3.1 GOOD_ALL | .705 / .0867 | .241 / .0341 | .079 / .0223 | .422 / .0527 |
| M3.2 BAD_ALL | .763 / .0774 | .265 / .0604 | .142 / .0340 | .442 / .0549 |
| M3.3 GOOD_TO_BAD | .687 / .0677 | .208 / .0452 | .104 / .0268 | .370 / .0524 |
| M3.4 BAD_TO_GOOD | .611 / .0902 | .292 / .0379 | .104 / .0265 | .382 / .0559 |

Question B history mean absolute q contributions are .0480/.0285/.0282/.0427; scene .0102/.0139/.0141/.0107; backlog .0000/.0140/.0051/.0067 (V11–V14, 99 predecessor contrasts each). Channel is zero in these sampled steps: none straddles a channel-input change. The separately included transition windows address those changes. Zero variation in a within-condition comparison is **not** evidence the actor ignores SNR.

Individual cases prevent a blanket ranking: under GOOD at frame 5503, history contributes +.2866 to a +.2871 q rise. At GOOD frame 6235, backlog contributes −.23733 to a −.2269 q fall (4/14 far hybrids). In the steady-BAD timeout case 1529, scene contributes +.0252, history −.0009. Sparse uniform samples miss some short events; the case cohort is not folded into these averages.

## 2. The transitions have different immediate responses

| Preceding → first new-channel decision | Observed q11 | Channel contribution | History contribution | Scene contribution | Far hybrids |
|---|---:|---:|---:|---:|---:|
| V13 f4326→4330, GOOD→BAD | .6359→.4782 | −.086817 | −.070867 | −.000017 | 4/14 |
| V14 f4168→4172, BAD→GOOD | .7095→.8080 | +.008850 | +.087100 | +.002550 | 8/14 |

Backlog contributes zero in both. In V13, prior MCS remains 27 at the first BAD decision; only SNR changes inside the channel block. History changes timeout→defined. The later high-q interval is not that immediate response: at f4366, history clears following timely undefined quality and contributes +.04875 versus scene −.01025. At f4386, action/presence return with timeout history, lowering q through history by .09595 while quality/latency remain unavailable.

In V14, the first GOOD decision also clears history after f4168's timely undefined quality; both SNR and MCS change. History supplies the largest contribution to the q increase, but 8/14 hybrid states fail the proximity diagnostic. At f4232, history returns and contributes −.0719 to the .8030→.7189 q fall. These are exact actor-function decompositions, not demonstrated interventions.

The earlier 20-decision payload medians remain observations: V13 93.39/68.64/99.62 kB and V14 79.00/56.93/95.54 kB, before/first-after/following. Payload is plotted, never attributed or inferred for swapped states.

## 3. “History” is explicit input, not recurrent memory

The intact 17-field block contains previous-mode one-hot, previous q, previous Q_perc, previous feedback latency, presence and success. Defined history supplies quality/latency; timeout retains action/presence but zeros unavailable quality/latency; cleared history zeros the entire block. A defined quality of zero is not missing quality. All four captures have one initial empty state; all 258 later cleared states follow recorded `TIMELY_UNDEFINED` outcomes. Counts and encodings are in [GLOSSARY.md](GLOSSARY.md).

A grouped contribution does not isolate previous quality. In the large GOOD rise, previous quality, q and latency all change. In the BAD timeout example, only previous q changes inside history, allowing that narrow interpretation. The V13 return-from-reset example narrows history to returning action/presence, not recovered quality. No state semantics were changed.

## 4. Mode 11 can remain preferred while q varies

Observed mode-11 (AE32/UINT4) selections are 1686/1771, 1489/1550, 1448/1533 and 1458/1529 (94.5–96.1%). Its representative mean margin stays positive, 2.301/2.089/2.242/2.059. The common references themselves already favour mode 11 on average (margin 1.938); attribution explains departures from that baseline, not why training established the preference.

The conditional q head is separate: representative q11 ranges span .2981–.8334, .2479–.9292, .2794–.8628 and .2186–.8992. Every explained non-11 target (15 total) also has its actual selected-mode margin and conditional q decomposed. At V11 f1521, mode changes 9→8: executed q falls .7146→.6707 while **q8 rises** .6547→.6707. Mode IDs are categories. Payload additionally depends on codec and input; no payload/latency/quality consequence is inferred from the actor-only queries.

## Admission, references and limitations

[PLAN.md](PLAN.md) and [selection JSON](selection_01/SELECTION.json) were frozen before attribution: 400 uniform targets plus the two existing 60-decision transition windows and 16 retained cases, 523 distinct targets. All 938 required recorded endpoints are admitted: 759 fresh exact replays plus 179 input/source-matched retained proofs, zero mode/q/state-hash/heads-hash mismatches.

Question A uses eight equally weighted real states: two per session, four GOOD/four BAD. Primary IDs are V11:442,1327; V12:387,1161; V13:349,1115; V14:1115,350 (zero-based decision indices). The predeclared alternative uses V11:221,1106; V12:193,968; V13:174,907; V14:1321,525. Exact frames/vectors/weights are recorded, not matched on history or scene. Primary history mix is 5 timeout/2 defined/1 cleared; alternative 3/4/1. Primary includes one 22,450-byte-backlog state; alternative backlogs are all zero. Neither is a training distribution.

Reference sensitivity is material: reference q11 changes .464025→.6398375, and margin 1.9383→2.2187. Scene remains largest under both backgrounds, but some secondary rankings and signed effects change; alternative GOOD margin contributions of scene/channel are almost tied (.581/.577). GOOD backlog q contribution becomes zero under the all-zero-backlog alternative. Do not promote these magnitudes into a universal ranking.

The unchanged diagnostic checks full-state block-balanced scaled distance with exact categorical/mask agreement to 6,383 recorded states (radius .09813645). It is **recorded-state proximity, not training or physical support**. Far nonendpoint slots are 38.2/40.6/37.1/38.8% under primary references and 29.6/29.6/27.9/29.1% under the alternative. No representative target has all 112 background hybrid slots near. Adjacent contrasts have all 14 slots near for 99/99, 89/99, 96/99 and 95/99 targets. No filtered subset is used in the summaries. All 16 coalitions stay in every exact decomposition; interaction allocation does not make hybrids physically realizable.

## Artifacts and closeout

The bounded attribution pass took 98.05 seconds: 7,903 unique reference–target pairs, 26,858 unique cached states (75 reused), CPU only; CUDA stayed uninitialized. Maximum additive error is 8.89×10⁻¹⁶. [Focused checks](checks_01/CHECKS.json) independently checked all 126,448 coalition slots, grouped aggregation, masks, sampling, hashes and all 16 retained cases; no transport/GT/launcher requalification.

- [Six final 3840×2160 PNGs and captions](CAPTIONS.md); [three recommended slides](SLIDES.md). All six final images were visually inspected. `figures_01` retains the initial layout draft; use `figures_02` for presentation.
- [Attribution CSV](attribution_01/ATTRIBUTIONS.csv), `ATTRIBUTIONS.json.gz`, `PAIR_COALITIONS.jsonl.gz`, `QUERY_CACHE.jsonl.gz`; complete per-decision, per-reference and support details.
- [Summary table](checks_01/SUMMARY.csv), [exact replay](attribution_01/EXACT_REPLAY.json), [results/input hashes](attribution_01/RESULTS.json).
- [Reproduction commands](COMMANDS.txt); `MANIFEST.json` binds final sources/outputs and disk accounting (excluding its own hash).

All new artifacts remain below 500 MiB; exact final size/headroom is in the manifest. No live services, GPU replay, training, runtime edits, registration, cleanup, deletion, archiving, commit or push. Existing attempts/verdicts are untouched. Consequences for performance, adaptive superiority and architecture success remain untested.
