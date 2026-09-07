# SplitFusion 72×4 UE campaign binding

The immutable 72-action SplitFusion catalog is bound to the canonical v2 four-profile network definition and the locked n78 100-MHz/273-PRB/μ=1/4D5U/5QI-6 radio baseline. The deterministic campaign inventory is 72 actions × 4 profiles = 288 unique cells; the unchanged representative pilot is 4 actions × 4 profiles = 16 unique cells.

- Binding: `splitfusion_72x4_campaign_binding_v1.json` (`85418003558823f8dc4a35e110112a20fb5e19fb2962f3dfa9362c1b0cb1ec32`)
- Catalog: `07e0690f8a55bdd6068b8b283d14b7e165ccbf44742dd0a9568cfdd5dcac54c3`, byte-identical to commit `4e237d719df91e83cfdc568e5f20b542d9482ad7`
- Radio: `fd604a37cfbc416412440dd447fe43cea9a94d3b65d69dd5e4e975de01a3dbf4`, byte-identical to commit `7a3e53d349f4de2ac71a6423e11188fadc1295f1`
- Campaign mapping: `add96c8f117c4cb8ba427c670ba1698d4a2b8aea84129b4d32ac47fdc56025b1`
- Pilot mapping: `e01565c640b180eeaa7e41d0024541860b4affd0cbdc80a544b7ec3cac12d1ad`
- Network profiles: `FAVORABLE_STABLE`, `MID_VARIABLE`, `ADVERSE_STABLE`, `FADE_RECOVERY`
- Pilot actions: 0 `split_noae_uint8_q0000`; 71 `split_ae32_uint4_q9800`; 46 `split_ae64_uint6_q9000`; 20 `split_ae128_uint8_q5000`

Real launch remains fail-closed. The retained SNR/RFsim mapping is qualified only for the legacy 40-MHz/106-PRB radio, and the retained launcher is explicitly superseded; neither can launch the 100-MHz/273-PRB campaign. This phase performs no RFsim recalibration.

The recorded, unexecuted localhost follow-up contains 36 SPLIT profiles (4 families × 3 quantizers × q {0, 0.30, 0.50}), 300 frames per profile, and front/encode-compress/transport/decode-decompress/back/end-to-end latency, payload, and delivery-rate measurements. LOCAL_CPU, LOCAL_GPU, and SKIP remain outside this contract.
