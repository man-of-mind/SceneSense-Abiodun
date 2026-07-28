# Draft note to advisor — split-inference motivation study (findings + a design question)

*(Draft for Abiodun to adapt/send. Honest framing of what the E1–E6 study found.)*

Hi [advisor],

I ran the motivation study we discussed (local processing/power, uplink, per-hop latency). It answered your questions,
and it also surfaced an architectural point I think we should decide on together before I write the motivation section.

**1. Local processing / power (your direct ask).** On the full model (no-AE, MobileNetV3 backbone + heads):
- The car running the *whole* model = **10.16 GMACs**; running only the front (backbone) under split = **2.45 GMACs**
  → split moves **76% of the FLOPs (the heads) to the edge**, a **4.15× cut in on-vehicle compute**.
- On GPU this is small in wall-clock/power (the model is tiny — full-local sustains real-time easily; power delta at
  10 FPS ≈ 1.5 W). On CPU (closer to a vehicle SoC) split cuts per-frame work ~2.2×.
- **E6 (compute-constrained sweep):** as we shrink the on-vehicle compute budget, full-local drops below the real-time
  deadline while split-front still meets it — I'll report the crossover point and cite where real automotive SoCs sit.

**2. The architectural point.** Our cooperative fusion is **detection-level (late)** — each ego produces detections
that we aggregate/triangulate in the spatial map. The split (front/back) is just a single-ego compute cut. That means
the "intermediate feature fusion beats late fusion" argument from the cooperative-perception literature **does not apply
to our current design** — so I don't think we can lean on it honestly.

**3. Honest consequence.** For detection-level fusion, running the model fully on each car and sharing detections gets
the **same cooperative result** as split, with **lower uplink, lower latency, and better privacy**. Split's one genuine
advantage over full-local is the **on-vehicle compute/power** in point 1 (the SWaP-C / thin-client angle). So the
defensible motivation is: *on compute-constrained vehicles that can't sustain real-time full-model inference, split is
what makes cooperative perception feasible* — plus our real contribution, **network-aware adaptation** of the feature
stream on a lossy 5G uplink.

**4. The fork I'd like your steer on.** If we wanted the stronger "feature-fusion" motivation, we'd have to actually
fuse multi-vehicle *features* at the edge (not detections). Two of our own results argue against that: the detection
head has been a dead-end (F1 ~0.35), and our one working cooperative result is **detection-level triangulation
(~1.40 m)**. So pivoting to feature fusion is real work with real risk. Options as I see them:
- (a) Keep detection-level fusion; motivate split on the **compute-constrained crossover (E6) + network-adaptation
  contribution**; treat feature sharing as a deployment mode, not "provably optimal."
- (b) Pivot to feature-level fusion to unlock the accuracy argument — larger scope, and our head/triangulation results
  suggest it may not pay off.

My recommendation is (a): it's honest, it uses our own numbers, and the network-aware controller is the novel part.
Happy to discuss.

— Abiodun
