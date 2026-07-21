# Codec comparison snapshot — zstd vs zlib at matched profiles (ideal 8 MB loopback, 100% delivery)

Same M', same action profile, only the entropy codec differs. Accuracy shown for **both** codecs to make the
lossless property explicit. Payload = entropy-coded test-set mean; front incl. compress; transport incl. decompress.

| Profile | mIoU zstd | mIoU zlib | loc m zstd | loc m zlib | payload KB zstd | payload KB zlib | front ms zstd | front ms zlib | transport ms zstd | transport ms zlib |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| AE-128 · u4 · ROI0.0 (~5%) | 0.819 | 0.819 | 0.88 | 0.88 | 129.2 | 127.4 | 26.8 | 26.7 | 2.6 | 5.6 |
| AE-64 · u8 · ROI0.3 (~7%) | 0.805 | 0.805 | 0.86 | 0.86 | 195.7 | 193.2 | 25.6 | 30.7 | 2.4 | 7.0 |
| no-AE · u6 · ROI0.0 (~28%) | 0.840 | 0.840 | 0.96 | 0.96 | 784.8 | 783.3 | 34.6 | 44.8 | 7.3 | 24.8 |
| no-AE · u8 · ROI0.0 (~1MB) | 0.840 | 0.840 | 0.95 | 0.95 | 1050.3 | 1052.9 | 27.8 | 46.0 | 8.7 | 30.7 |

- **mIoU and loc are identical zstd vs zlib at every profile** → entropy coding is lossless (decoded tensor unchanged).
- Payload within ~2% (compression-ratio difference). Transport: zstd ~2× lower at small payloads, ~3.5× lower at ~1 MB.
