# UE downlink SNR versus gNB received uplink PUSCH SNR — bridge qualification

**Verdict: `NOT_SUPPORTED_AS_UPLINK_PREDICTOR`** (4 of 7 registered checks passed)

**Scope of that verdict:** it applies to `UE_PHY_MEAS.snr` *as produced by this RFsim testbed under the
registered uplink-only channel actuation*. It is **not** a finding that a UE-side downlink SNR is
uninformative in general, and it is **not** a claim about real radio hardware. The root cause is
identified below and is structural, not statistical.

Evidence: `rl_agent/experiments/ue_snr_bridge_qualification_v1/20260923_220340`
(`analysis_v1.json`, `manifest.json` with SHA-256 for every file, `figures/`).

---

## 1. What was asked and what was measured

The question was never whether the two quantities are numerically equal — they measure opposite link
directions. It was whether the UE-side signal is **available before decisions, fresh enough, correctly
ordered across network conditions, responsive to the controlled channel, and useful** as the policy's
physical-channel input.

Two labels are used throughout and are never collapsed into a bare "SNR":

| Quantity | Direction | Observer | Units |
|---|---|---|---|
| `UE_PHY_MEAS.snr` — *UE downlink SNR* | UE receive / downlink | UE | plain integer dB (**not** ×10) |
| `GNB_MAC_PUSCH_POWER_CONTROL.snrx10` — *gNB received uplink PUSCH SNR* | gNB receive / uplink | gNB | dB ×10 (divided by 10; conversion recorded) |

## 2. Headline result

The controlled channel moved the **uplink** exactly as designed, and the UE's **downlink** measurement
did not move at all.

| Profile | gNB uplink PUSCH SNR P50 | UE downlink SNR P50 | UL throughput | UL loss | UL jitter |
|---|---:|---:|---:|---:|---:|
| FAVORABLE_STABLE | **16.50 dB** | −93.0 dB | 4.000 Mbps | 0.00 % | 3.678 ms |
| FADE_RECOVERY | **15.00 dB** | −93.0 dB | 4.000 Mbps | 0.00 % | 3.175 ms |
| MID_VARIABLE | **14.50 dB** | −93.0 dB | 4.000 Mbps | 0.00 % | 3.398 ms |
| ADVERSE_STABLE | **8.50 dB** | −93.0 dB | 4.000 Mbps | 0.00 % | 2.636 ms |

The gNB ordering is the registered ordering. The UE column is constant.

## 3. Root cause — why the UE column is flat

Two independent, verified reasons. Both are properties of the harness, not of the statistics.

**(a) The registered actuator impairs the uplink only.** `channel_state_initial.txt` shows three channel
models on the gNB, of which exactly one — `rfsimu_channel_ue0` — has `model owner: rfsimulator`; the
`rfsimu_channel_enB0` / `enB1` entries are unowned placeholders. The actuator commands
`rfsimu_channel_ue0`, and the measured response appears in the gNB's *received uplink* PUSCH SNR. The
downlink path the UE measures is never impaired, so there is no downlink condition for the UE to track.

**(b) The UE downlink measurement path is degenerate in this build.** From the 3,397 retained rows:

| Field | Range | Distinct values |
|---|---|---:|
| `rsrp` | −2147483648 (int32 min) in **100 %** of rows | 1 |
| `rssi` | −183, constant | 1 |
| `rx_power` | −90, constant | 1 |
| `noise_power` | −90 … 8 | 7 |
| `snr` | −98 … 0 | 7 |
| `w_cqi` | −98 … 0 | 7 |

`snr == rx_power − noise_power` in **3,397/3,397** rows, so the emitted SNR varies *only* through
`noise_power` and takes 7 distinct values overall. A value of −93 dB is not a physically meaningful SNR
(the link simultaneously carried 4 Mbps at high MCS with zero loss); these are uncalibrated internal
fixed-point magnitudes, not a usable receive SNR.

**Incidental confirmation of the Phase-A source reading:** `w_cqi == snr` in **3,397/3,397** rows. That is
exactly what `nr_ue_measurements.c:102` and `phy_procedures_nr_ue.c:415` predict — the two fields are the
same expression. `w_cqi` is therefore not a standardized CQI index and adds no independent signal.

## 4. Check-by-check

| Check | Result | Evidence |
|---|---|---|
| Fresh UE coverage at 10 Hz | **PASS** | worst per-profile 100 ms bin coverage **0.9357**; 0.9357 / 0.9478 / 0.9478 / 0.9880 |
| Sufficient UE samples | **PASS** | 557 / 549 / 842 / 607 per profile |
| FAVORABLE above ADVERSE ordering | **FAIL** | both P50 = −93.0 dB; Cliff's δ = **−0.0165** (NEGLIGIBLE); distribution overlap **99.93 %** |
| Useful positive association with uplink | **FAIL** | pooled Spearman **−0.0135**, 95 % CI **[−0.0625, +0.0368]** (spans zero); every per-profile CI also spans zero |
| No fabricated / forward-filled measurement | **PASS** | no forward fill; 20/20/90/30 unmatched UE samples dropped, never carried |
| No reliance on future samples | **PASS** | availability uses only observations at or before each boundary |
| UE receive path produces a live measurement | **FAIL** | degenerate: RSRP unpopulated 100 %, RSSI and `rx_power` constant |

**Availability is genuinely good — informativeness is absent.** That separation is the useful part of this
result: the signal *arrives* fresh enough for a 10 Hz policy, it simply carries nothing here.

## 5. Association, ordering and tracking

- **Per-profile Spearman** (95 % moving-block bootstrap CI, block ≈ 1 s of UE samples):
  FAVORABLE −0.031 [−0.105, +0.053]; MID +0.025 [−0.083, +0.127];
  ADVERSE −0.002 [−0.067, +0.081]; FADE −0.076 [−0.155, +0.008]. **All four span zero.**
- **Pooled**: Pearson −0.0089, Spearman −0.0135, CI [−0.0625, +0.0368], 2,395 pairs.
- **Temporal tracking (FADE_RECOVERY)**: cross-correlation peak r = **0.054** at lag 18 pairs. The
  analyzer labels this `UE_DOWNLINK_SNR_LEADS_...`, but at r = 0.054 that is **indistinguishable from no
  tracking** and must not be reported as a lead/lag finding.
- **UE SNR vs uplink throughput/loss/jitter**: the correlation is **undefined, not zero** — all four UE
  P50 values are identical, so the coefficient has no variance to work with. It is reported as missing
  rather than fabricated as 0.

## 6. Time alignment

Both softmodems run on one host and every T event is stamped `CLOCK_REALTIME` at its call site, so the two
tracers share a clock domain. The CSV sink drops the date and timezone, so a recorded same-host wall/monotonic
anchor supplies them; the nearest-day candidate is chosen, which also absorbs a midnight rollover.

Pairing is nearest-neighbour inside ±5 ms (half the 10 ms UE emission period). Measured quality:
pair distance **P50 0.51 ms / P95 3.94 ms**, **0 ambiguous ties** in every profile, and no negative
intervals. Unmatched UE samples are dropped. **No forward fill anywhere.** Observation age at decision
boundaries: P50 14.5–24.0 ms, P95 52.2–105.8 ms. UE measurement cadence 22.0–33.8 Hz, consistent with the
`nr_slot_rx == 0` gate (≤ 100 Hz) reduced by PDSCH availability.

## 7. Traffic and profile fidelity

One constant offered load for all four profiles: **uplink 4.0 Mbps, downlink 1.0 Mbps, 1200 B UDP
datagrams**, two concurrent one-way iperf3 sessions (not `--bidir`, which would force one rate on both
directions). All four profiles: **300/300 registered samples, 0 skipped, 0 clamped**, 262–270 distinct
noise commands each.

The eight registered mapping anchors stop at 19.5 dB achieved, below the FAVORABLE targets, so the upper
anchor was **measured live before any replay**: commanded −12.5 dB → **25.0 dB** achieved median PUSCH SNR
over 1,826 samples. Nothing was extrapolated; the zero clamp count confirms full coverage.

## 8. Limitations

1. **The offered load did not reach the delivery frontier.** 4 Mbps was carried at 4.000 Mbps with 0.00 %
   loss in *every* profile, including ADVERSE_STABLE at 8.5 dB. The rate was chosen not to saturate, but it
   was low enough that uplink delivery could not discriminate the profiles either. A higher offered load
   would be needed to probe degradation — though it would not change the UE-side conclusion, which is
   driven by the flat measurement path.
2. **Uplink-only actuation.** This is the decisive limitation and is a property of the registered harness.
3. **RFsim, not hardware.** The degenerate RSRP/RSSI/`rx_power` path is a simulator artifact.
4. **Single UE, single host, one repetition per profile**, ~25 s measured per profile. A short
   qualification, not a campaign.
5. **Association is not causation**, and the profile-level delivery relation rests on four points.

## 9. What this does and does not authorize

- **Do not** adopt `UE_PHY_MEAS.snr` as the SplitFusion policy's physical-channel feature on this evidence.
- **Do not** conclude that a UE-side downlink channel feature is unusable in principle. That was not tested
  and could not be, because the downlink was never impaired.
- **The actionable finding for the project** is the harness limitation: the registered network-profile
  actuator moves the uplink channel model only. Any future downlink-derived policy feature needs the
  actuator extended to the UE-side channel model, and needs a UE receive path that actually populates
  RSRP/RSSI — otherwise the feature is untestable regardless of how the policy is designed.
- The **gNB uplink PUSCH SNR remains the only measured, responsive channel quantity** in this testbed, and
  it is not available to the UE at runtime. That gap is unchanged by this study.

## 10. Integrity and teardown

Channel restored to the registered cold value and read back: commanded **−50.0 dB**, read back **−50.0 dB**,
verified. Final host state **cold**: no orphan softmodem, tracer, or iperf3 process (host or container), no
residual `oaitun_ue1`.

Every figure was produced in **both PNG and vector PDF** (8 files). Two artifact classes are retained on
disk and hashed in `manifest.json` but **not committed**, because `.gitignore` excludes them (`*.raw`
line 42, `*.pdf` line 67) and no force-add was authorized: the raw T-tracer captures
(`ttracer/{gnb,ue}/*.raw`, 49 MB) and the vector PDF figures. The PNG figures and every extracted CSV are
committed.
