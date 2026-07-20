# Bounded/default-buffer loopback calibration

Clean run batch: `calib2_clean_20260718_bounded`

Command:

```bash
env CONFIRM_SYSCTL=1 FPS_LIST=10 DURATION_S=10 BATCH_ID=calib2_clean_20260718_bounded \
  bash downlink_latency_fps/run_bounded_loopback_fps.sh
```

Purpose: check whether the historical/default UDP receive-buffer condition can support the current no-AE 200k payload before spending time on a full FPS sweep.

## Result

| condition | FPS | frames | received | delivery | feature KB p50 | feature chunks | result KB p50, delivered only | RTT-recv p50 ms, delivered only | tail-done→recv p50 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded loopback 208 KB | 10 | 100 | 3 | 0.030 | 1095.6 | 19 | 15.9 | 56.2 | 6.1 |

Delivered-only latency split:

| condition | FPS | delivery | front ms | uplink payload-handling ms | back ms | downlink ms | post-send RTT ms | capture→result est ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bounded loopback 208 KB | 10 | 0.030 | 45.1 | 32.3 | 17.8 | 6.1 | 56.2 | 106.8 |

Do not over-interpret the bounded latency split: it is based on only three delivered frames after many failed feature
bursts. The robust result is the delivery failure itself.

Delivered rows from the clean run:

| delivered frame order | front ms | feature payload KB | chunks | back ms | downlink ms | post-send RTT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 39.3 | 1079.8 | 19 | 386.6 | 6.1 | 424.9 |
| 2 | 50.6 | 1102.0 | 19 | 17.8 | 6.1 | 56.2 |
| 3 | 44.8 | 1093.2 | 19 | 8.7 | 5.4 | 46.7 |

The large first `back_ms` is not representative steady edge compute. It is a sparse-delivery first-success /
warm-up outlier that dominates the original short calibration because almost no frames returned. Once the first
successful frame is past, `back_ms` returns to the expected ~9–18 ms range.

## Interpretation

The bounded/default-buffer loopback condition is not a clean `~50 ms` transport point for the current no-AE 200k recipe. It reproduces the UDP delivery cliff:

- no-AE feature payload is about `1.1 MB`;
- the burst is about `19` UDP chunks at `60 KB` each;
- with the old `net.core.rmem_max=212992` cap, the effective receive buffer is only about `416 KB`, or roughly `7` chunks;
- losing one chunk prevents reassembly, so most frames never return.

Therefore a full bounded-loopback no-AE FPS sweep would mostly measure packet/drop failure, not clean downlink latency. Keep this condition for Step 2 reliability/buffer experiments, not Step 1 downlink characterization.

## Buffer regime — what this does and does NOT show (important, do not carry the wrong takeaway)

This is a **buffer-cliff demonstration, not a "loopback is broken" result.** The delivery number is entirely a
function of the UDP receive-buffer size for the same `1.1 MB / 19-chunk` no-AE 200k payload:

| condition | `rmem_max` | delivery (same payload) |
|---|---:|---:|
| **8 MB-buffer loopback** (all Stage-1 staleness/latency work; the ideal FPS sweep) | `8388608` | **100%** |
| default-buffer loopback (this calibration) | `212992` | 1–3% |
| default-buffer loopback (historical `PPS_DEPLOY_RESULTS.md`, pre-fix) | `212992` | ~18.8% (75/400) |

Concretely: this session's own `20260716_213618_front_fusion_ego_25` loopback run delivered **300/300 = 100%** with
the identical `~1088 KB / 19-chunk` payload, and the ideal FPS sweep delivered 100% at every FPS. So:

- **Nothing regressed in the model or back-half.** Delivered frames return in the expected `~47–56 ms` post-send RTT
  the moment a payload gets through; the failure is purely feature-payload reassembly under a too-small buffer.
- The "decent loopback" from earlier work was — and still is — the **8 MB-buffer** condition. Raising the buffer during
  the knob-matrix work is exactly *why* those runs were clean. Default-buffer loopback was never the condition our good
  Stage-1 results came from.
- The `PPS_DEPLOY` ~18.8% is a **pre-buffer-fix default-buffer** number, i.e. the same cliff, not evidence loopback is
  inherently lossy. The 3%-vs-18.8% gap is short-run/high-FPS burst variance in a rare-delivery regime, not breakage.

**Takeaway:** default-buffer loopback cannot hold a 19-chunk burst → that is why we set 8 MB. The `8 MB-buffer` loopback
is the Step-1 latency/downlink anchor (100% delivery); the default-buffer number belongs only in Step-2 buffer/reliability
discussion as a cliff artifact.

This clean 3/100 result should replace the earlier 1/100 number for reporting. The old project notes still matter:
`PPS_DEPLOY_RESULTS.md` recorded 75/400 returns for the 200k no-AE bounded loopback case (~18.8%), and
`rl_agent/TRANSPORT_CONFIG.md` recorded 0.11–0.13 delivery for u8-sized payloads. So the current result is worse
than the historical bounded runs, but it is the same failure mode rather than a new model/back-half behavior.
Likely contributors are short-run variance in a rare-delivery regime, the current full moving-ego staleness harness
and scheduling load, and exact socket/sysctl state not perfectly matching the old runs.

## Restore / safety note

The first calibration (`calib_20260717_bounded`) delivered 1/100 frames and exposed two hygiene issues: a stale
local back-half could occupy the UDP port, and `run_common.sh` overwrote the bounded script's restore trap, leaving
`net.core.rmem_max/wmem_max` at `212992`. This was immediately corrected manually:

```bash
sudo sysctl -w net.core.rmem_max=8388608 net.core.wmem_max=8388608
```

The scripts were patched so `run_common.sh` now calls an optional `after_run_common` cleanup hook, and `run_bounded_loopback_fps.sh` restores sysctls through that hook.
`run_common.sh` was also patched to refuse local back-half startup when the UDP back-half port is already occupied,
so stale listeners cannot silently mask launch failures.

Post-fix verification:

```text
net.core.rmem_max = 8388608
net.core.wmem_max = 8388608
SO_RCVBUF after requesting 8 MiB = 16777216
```
