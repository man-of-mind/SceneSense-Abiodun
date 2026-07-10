# Loopback transport config + the delivery-cliff mechanism (lock this in for the presentation)

## The result to explain
Loopback delivery rate collapses with payload size:
| payload | chunks/frame (60KB each) | delivery | back-half compute |
|---|---|---|---|
| 366–385 KB (u4) | 7 | **1.00** | 7 ms |
| 717–761 KB (u4-none, u6) | 13 | 0.12–0.32 | 9 ms |
| 983–1426 KB (u8) | 17–25 | 0.11–0.13 | 7–8 ms |

## The mechanism (measured, not assumed)
It is **UDP datagram loss from a bounded receive buffer**, NOT the timeout and NOT bandwidth:
- Delivered frames return in **~39 ms median** (max 224 ms) — far under the `--result-timeout 1.5 s`. Failed
  frames have `result_received=False` and never arrive → a longer timeout recovers zero of them.
- Back-half compute is ~8 ms → the receiver is not compute-lagging.
- Features are split into **60 KB UDP datagrams** (`--chunk-bytes 60000`, 8-byte header `!IHH` =
  msg_id/chunk_idx/total_chunks) and sent as a **tight burst, no pacing** (`for chunk: sendto`).
- UDP has no retransmission. A burst that exceeds the receive buffer drops chunks; losing ONE chunk means
  the feature can't be reassembled → frame lost.

## The config that MATTERS (and the gotcha)
| knob | value | note |
|---|---|---|
| chunk_bytes | 60000 B | one UDP datagram/chunk (payload 59992 B + 8B header) |
| header | 8 B (`!IHH`) | msg_id (u32), chunk_index (u16), total_chunks (u16) |
| SO_RCVBUF | **requested 8 MB, GRANTED 416 KB** | kernel caps at `net.core.rmem_max` |
| **net.core.rmem_max** | **212992 B (208 KB)** | THE cap. Needs root to raise. |
| SO_SNDBUF | OS default (not set) | `net.core.wmem_max` also 208 KB |
| result_timeout | 1.5 s | NOT the gate (delivered RTT ~39 ms) |
| effective rcv buffer | ~416 KB ≈ **7 × 60 KB chunks** | matches the cliff exactly |

**Gotcha / integrity point:** the code requests `SO_RCVBUF=8MB`, but `net.core.rmem_max=208KB` silently
caps it to 416 KB (verified: `getsockopt` returns 425984 after requesting 8 MB). So the delivery cliff at
~7 chunks is set by this **208 KB cap + 60 KB chunks**, i.e. it is partly a **transport-config artifact**,
not a fundamental limit. If `rmem_max` were raised to the intended 8 MB, the buffer would hold ~140 chunks
and large payloads would deliver on loopback.

## How to present it honestly
- The **mechanism is real and general**: payload → datagram count → loss under a bounded receive buffer,
  no retransmission. Compression cuts datagrams → restores reliability. That motivates the control policy.
- But state the buffer config explicitly (above) so the cliff location is **reproducible and explained** —
  don't present it as a fundamental property of "1 MB payloads."
- The **fundamental** reliability constraint (bandwidth, RF loss, latency) comes from the **real channel
  (OAI/Sionna)** — that is the phase that gives the non-artifact reliability numbers.

## DECISION (2026-07-09): Option B — ideal transport for Month 2
Raised `net.core.rmem_max` and `net.core.wmem_max` to 8388608 (8 MB) via `sudo sysctl -w` (RUNTIME change,
NOT persistent across reboot — to persist, add to /etc/sysctl.conf). A UDP socket now gets ~16 MB granted
(~140 x 60 KB chunks), so all our <=1.4 MB payloads deliver ~100%. **Month-2 loopback is now an IDEAL local
transport: 8 MB buffers, NO bandwidth cap / no Linux `tc` shaping.** Present it as such. Month-2 reports
accuracy + payload + latency (front=UE compute, back=edge compute, transport=localhost round-trip).
Reliability + latency under a REAL channel (bandwidth, RF loss) = OAI + Sionna (Month 3).
NB: latency numbers must be RE-MEASURED under the raised buffers (the earlier loopback ran at 208 KB); the
matrix currently carries placeholder latency from that run pending the ideal re-run.

## Options (historical — Option B was taken)
- **A. Present loopback as "bounded-buffer transport stress"** with the config locked in (honest, no root
  needed). Reliability-under-real-channel = OAI phase.
- **B. Raise `net.core.rmem_max` to 8 MB** (sysctl, needs root — shared box, get supervisor OK) and re-run:
  loopback becomes an ~ideal transport (delivery ~100%), isolating latency; real loss comes only from OAI.
  Cleaner separation of "ideal transport" vs "real channel," but requires a system change.
