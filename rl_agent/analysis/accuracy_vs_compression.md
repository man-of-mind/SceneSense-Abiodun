# Accuracy vs compression (offline, deterministic — 200k fusion model, test split)

Model run through the split-inference codec round-trip at each quant profile; uncompressed baseline routes the model directly. Same eval thresholds (thr 0.10, nms 6, ≤40 m). All numbers directly comparable (same code path).

| profile | mIoU | vehicle IoU | person IoU | obj recall | veh recall | ped recall | veh loc MAE (m) | ped loc MAE (m) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.837 | 0.934 | 0.579 | 0.775 | 0.799 | 0.734 | 1.073 | 1.379 |
| q_pchan_u8_zlib | 0.837 | 0.934 | 0.58 | 0.775 | 0.799 | 0.732 | 1.076 | 1.377 |
| q_pchan_u8_none | 0.837 | 0.934 | 0.58 | 0.775 | 0.799 | 0.732 | 1.076 | 1.377 |
| q_ptensor_u8_zlib | 0.837 | 0.934 | 0.579 | 0.742 | 0.782 | 0.673 | 1.144 | 1.426 |
| q_ptensor_u8_none | 0.837 | 0.934 | 0.579 | 0.742 | 0.782 | 0.673 | 1.144 | 1.426 |
| q_pchan_u6_zlib | 0.837 | 0.934 | 0.58 | 0.773 | 0.797 | 0.731 | 1.083 | 1.394 |
| q_pchan_u6_none | 0.837 | 0.934 | 0.58 | 0.773 | 0.797 | 0.731 | 1.083 | 1.394 |
| q_pchan_u4_zlib | 0.836 | 0.934 | 0.576 | 0.77 | 0.799 | 0.72 | 1.125 | 1.422 |
| q_pchan_u4_none | 0.836 | 0.934 | 0.576 | 0.77 | 0.799 | 0.72 | 1.125 | 1.422 |
