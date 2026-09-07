# Phase 13C 36-profile × 300-frame localhost replay measurement

Status: `SPLITFUSION_PHASE13C_36X300_LOCALHOST_MEASUREMENT_COMPLETE`.

This is a live CUDA/localhost software replay measurement. Sensor capture and disk input loading are excluded. Localhost timing is not OAI, RFsim, radio, or Raspberry Pi latency.

The immutable Phase-4 fit sample is `ad2cb25ea8d64d8d774c9c455c09e169f9a3eb1b51bf6143e82dfacdecf10ee9`; the preregistered rotated schedule is `68915d1ce18a312720b93e042d967a473ced17987b99b9f1a10d74ddeacf2cbc`.

Perception accuracy was not rescored. No holdout, validation, or test sensor frame and no box, semantic-GT, AVO, depth-GT, or evaluation record was opened.

| action | family | quantizer | q | complete | E2E median ms | E2E p95 ms | inner median B | SFD1 median B | on-wire median B | datagrams median | seq. fps |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | noAE | UINT8 | 0.00 | 300 | 176.551 | 220.711 | 3573199 | 3573484 | 3583816 | 287 | 5.672 |
| 1 | noAE | UINT8 | 0.30 | 300 | 170.748 | 221.539 | 2515164 | 2515449 | 2522721 | 202 | 5.859 |
| 2 | noAE | UINT8 | 0.50 | 300 | 157.042 | 206.321 | 1790391 | 1790676 | 1795860 | 144 | 6.388 |
| 6 | noAE | UINT6 | 0.00 | 300 | 190.217 | 243.649 | 2743761 | 2744046 | 2751966 | 220 | 5.211 |
| 7 | noAE | UINT6 | 0.30 | 300 | 174.711 | 222.076 | 1953773 | 1954058 | 1959710 | 157 | 5.639 |
| 8 | noAE | UINT6 | 0.50 | 300 | 159.540 | 213.117 | 1396448 | 1396737 | 1400769 | 112 | 6.223 |
| 12 | noAE | UINT4 | 0.00 | 300 | 179.519 | 229.344 | 1477792 | 1478077 | 1482361 | 119 | 5.571 |
| 13 | noAE | UINT4 | 0.30 | 300 | 162.111 | 216.575 | 1095195 | 1095480 | 1098648 | 88 | 6.127 |
| 14 | noAE | UINT4 | 0.50 | 300 | 152.527 | 203.244 | 799577 | 799862 | 802202 | 65 | 6.513 |
| 18 | AE128 | UINT8 | 0.00 | 300 | 154.128 | 215.216 | 2344795 | 2345081 | 2351849 | 188 | 6.405 |
| 19 | AE128 | UINT8 | 0.30 | 300 | 142.651 | 191.266 | 1672043 | 1672329 | 1677153 | 134 | 6.934 |
| 20 | AE128 | UINT8 | 0.50 | 300 | 129.471 | 176.272 | 1208614 | 1208900 | 1212392 | 97 | 7.622 |
| 24 | AE128 | UINT6 | 0.00 | 300 | 162.159 | 208.722 | 1866423 | 1866709 | 1872109 | 150 | 6.133 |
| 25 | AE128 | UINT6 | 0.30 | 300 | 149.692 | 202.165 | 1337295 | 1337585 | 1341473 | 108 | 6.632 |
| 26 | AE128 | UINT6 | 0.50 | 300 | 134.882 | 191.450 | 964555 | 964841 | 967649 | 78 | 7.245 |
| 30 | AE128 | UINT4 | 0.00 | 300 | 153.473 | 201.427 | 903844 | 904130 | 906758 | 73 | 6.401 |
| 31 | AE128 | UINT4 | 0.30 | 300 | 130.118 | 175.340 | 673447 | 673733 | 675677 | 54 | 7.525 |
| 32 | AE128 | UINT4 | 0.50 | 300 | 121.076 | 165.262 | 498667 | 498953 | 500393 | 40 | 8.146 |
| 36 | AE64 | UINT8 | 0.00 | 300 | 132.046 | 181.769 | 1217455 | 1217740 | 1221268 | 98 | 7.400 |
| 37 | AE64 | UINT8 | 0.30 | 300 | 120.610 | 165.777 | 862111 | 862396 | 864916 | 70 | 8.147 |
| 38 | AE64 | UINT8 | 0.50 | 300 | 116.702 | 160.657 | 619525 | 619810 | 621610 | 50 | 8.382 |
| 42 | AE64 | UINT6 | 0.00 | 300 | 144.522 | 184.362 | 958419 | 958704 | 961476 | 77 | 6.987 |
| 43 | AE64 | UINT6 | 0.30 | 300 | 123.797 | 166.307 | 682686 | 682971 | 684951 | 55 | 8.047 |
| 44 | AE64 | UINT6 | 0.50 | 300 | 118.416 | 162.046 | 490902 | 491191 | 492631 | 40 | 8.307 |
| 48 | AE64 | UINT4 | 0.00 | 300 | 125.042 | 177.295 | 482832 | 483117 | 484521 | 39 | 7.759 |
| 49 | AE64 | UINT4 | 0.30 | 300 | 118.465 | 159.974 | 359097 | 359382 | 360426 | 29 | 8.376 |
| 50 | AE64 | UINT4 | 0.50 | 300 | 118.002 | 157.256 | 262907 | 263193 | 263985 | 22 | 8.441 |
| 54 | AE32 | UINT8 | 0.00 | 300 | 119.302 | 156.574 | 602021 | 602306 | 604070 | 49 | 8.383 |
| 55 | AE32 | UINT8 | 0.30 | 300 | 116.962 | 154.591 | 428083 | 428371 | 429631 | 35 | 8.467 |
| 56 | AE32 | UINT8 | 0.50 | 300 | 119.127 | 165.753 | 308430 | 308715 | 309615 | 25 | 8.233 |
| 60 | AE32 | UINT6 | 0.00 | 300 | 124.154 | 169.653 | 474229 | 474514 | 475882 | 38 | 7.944 |
| 61 | AE32 | UINT6 | 0.30 | 300 | 118.022 | 157.015 | 339225 | 339510 | 340518 | 28 | 8.359 |
| 62 | AE32 | UINT6 | 0.50 | 300 | 115.150 | 155.180 | 244612 | 244899 | 245619 | 20 | 8.562 |
| 66 | AE32 | UINT4 | 0.00 | 300 | 120.230 | 155.349 | 235128 | 235413 | 236097 | 19 | 8.306 |
| 67 | AE32 | UINT4 | 0.30 | 300 | 117.389 | 154.785 | 175674 | 175959 | 176499 | 15 | 8.499 |
| 68 | AE32 | UINT4 | 0.50 | 300 | 114.768 | 159.380 | 129540 | 129825 | 130221 | 11 | 8.540 |

All 10,800 transactions completed with exact reassembly, one compression, one decompression and one frozen tail call each. Frozen perception, ranker, and AE states were unchanged.

Component timings are diagnostic boundaries. `localhost_delivery_ms` overlaps the send portion of `sfd1_fragment_send_ms`; no double-counted component sum is presented. End-to-end time is measured independently from frame-context construction through serialized edge return.

Remaining blockers are 100-MHz RFsim calibration and the 16-cell OAI pilot. The 288-cell OAI campaign was not launched.
