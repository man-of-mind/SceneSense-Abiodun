# Capacity Qualification Retry Amendment: MTU-Safe Packetization V1

Status: `REGISTERED_NOT_LAUNCHED`

This is a narrow, additive amendment to the bounded capacity qualification. It does not replace the original preregistration, change the action catalogue, alter the production SplitFusion sender, or modify the preserved refused attempt.

## Why a retry is required

The preserved attempt `20260924_234501` passed its achieved-PUSCH-SNR, saturation-backlog, cross-layer-service, lifecycle, and cold-state checks, but measured zero P50 application goodput at all three points. The evidence is:

| Point | Target SNR | Achieved PUSCH SNR P50 | Backlogged fraction | gNB PDCP bytes | Complete chunks at ext-DN | Application-goodput P50 |
|---|---:|---:|---:|---:|---:|---:|
| p25 | 7.827 dB | 8.0 dB | 1.0 | 36,605,364 | 54 | 0 Mbps |
| p50 | 8.608 dB | 8.5 dB | 1.0 | 38,234,700 | 118 | 0 Mbps |
| p75 | 9.604 dB | 9.5 dB | 1.0 | 40,712,948 | 191 | 0 Mbps |

The old capacity probe put a 24-byte `SSBURST` header in front of each 60,000-byte payload chunk. That produced a 60,024-byte UDP payload and a 60,052-byte IPv4 packet, requiring about 41 IPv4 fragments on a 1,500-byte MTU path. Under overload, loss of any fragment prevents delivery of the complete UDP datagram. The retained PDCP delivery with almost no complete ext-DN datagrams makes fragmentation a concrete methodological confound strongly consistent with the refusal. It is not claimed as the proven sole cause.

Preserved refusal bindings:

- `CAPACITY_QUALIFICATION_RESULT.json`: `5f838f3d3a3f70b4d0b5510c01d2a0af4aed80a47f1efeed3334953ea3314444`
- `manifest.json`: `d86c898f24738ca3db567237ef0cae393668da1cd8ac28de48331488f1983197`
- `CAPACITY_QUALIFICATION_REFUSED.json`: `de565c0ab35630d3db8b323cdeee0c5502603a7ec36dd0e63a8aaa331b1a53f1`

These hashes identify the preserved evidence; this amendment does not rewrite that attempt.

## Registered retry change

Only the capacity probe uses 1,200-byte payload chunks. With the 24-byte `SSBURST` header, the UDP payload is 1,224 bytes and the complete IPv4 packet is 1,252 bytes, below the 1,500-byte path MTU. One exact 3,568,326-byte frame becomes 2,974 chunks: 2,973 full 1,200-byte chunks and one 726-byte tail chunk.

The following remain unchanged:

- action 0 / `split_noae_uint8_q0000`;
- 3,568,326 application payload bytes per frame;
- 285.46608 Mbps offered application rate;
- 10 Hz frame cadence and the registered point durations;
- the production scientific `CHUNK_BYTES=60000` path.

The custom capacity sink derives its chunk bound from the registered frame and chunk sizes, validates every full and tail payload, and therefore supports the required chunk indexes above the shared production receiver's 1,024-chunk limit. Every sender, sink, point, and top-level result records the same packetization identity; the offline verifier reconstructs and checks it.

## Bounded-load caveat and gates

The retry offers 386,620 datagrams per point (about 29,740 datagrams/s over the 13-second point). This is intentionally a saturating probe, not a production traffic pattern. Exact sender handed/dropped accounting, packetization-mismatch counters, primary ext-DN bins, achieved-SNR checks, persistent-backlog checks, post-point ingress-quiet drain proof, teardown, and final cold-state gates remain mandatory. A packet-processing bottleneck can still make the retry refuse; no scientific threshold is relaxed.

