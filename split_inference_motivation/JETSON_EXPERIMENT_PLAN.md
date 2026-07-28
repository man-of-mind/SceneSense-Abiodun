# Embedded-GPU (Jetson) experiment — plan for a new session

**Why this exists.** The compute-crossover result (E6) is a **CPU-arm** result. The **GPU arm** (fig8) throttled an
RTX 5090 from 2872→210 MHz and found **no crossover** — even 13× throttled it holds 55 FPS. But a *throttled
datacenter GPU is still far stronger than a real vehicle GPU*, so that does **not** answer "does a car *with a GPU*
need split?". Only a real embedded GPU can. This experiment measures full-local vs split on an **NVIDIA Jetson Orin**.

**⚠️ Needs hardware.** Requires a physical Jetson Orin (Nano / NX / AGX). If none is available, this experiment is
**blocked** — do not fake it. Fall back to the honest position already in the deck: the extrapolation (fig6) + the
GPU-arm mechanism (fig8) + cited Jetson specs, stating the embedded-GPU crossover is *unmeasured*.

## Goal / question
Does running the whole model locally on a **Jetson-class car GPU** miss the real-time deadline while split-front meets
it — i.e. is there a GPU crossover on real embedded hardware, at what power mode, and how does energy compare?

## Method (mirror E1 + E6 on the Jetson)
1. **Get the model onto the Jetson.** Export the no-AE checkpoint
   (`experiments/ae_integrated_20260710/noae_baseline/checkpoints/mprime_joint_noae/best.pt`) + the model code.
   Use the same real 7-channel input tensor (`1×7×432×768`) as E1 (ship one saved sample; do not require CARLA on
   the Jetson). Verify FRONT+BACK == FULL bit-exact, as E1 did, before profiling.
2. **Resource + throughput (E1/E6 style), for FULL vs FRONT (split):**
   - Latency (p50/p95) and **sustained FPS** over a 20 s pinned window, on the Jetson **GPU** and **CPU**.
   - Sweep the Jetson's real power envelopes with `nvpmodel` (e.g. Orin NX: 10 W / 15 W / 25 W) and `jetson_clocks`
     on/off. This is the *proper* SWaP-C axis — real power modes, not an artificial clock throttle.
3. **Power / energy — measured, not derived.** Jetson exposes real on-board rail power. Sample with `tegrastats`
   (or `jtop`/`jetson-stats`) during each sustained run → actual Watts and **energy/frame (J)** for FULL vs FRONT.
   This finally gives *transferable absolute watts* that E2 could not (E2 only licensed the % ratio off an RTX 5090).
4. **(If time) repeat at a heavier model** (larger backbone or a second perception model) to place a real anchor on
   fig6's extrapolation — the single most valuable add for the paper.

## Metrics to report (per power mode)
`nvpmodel | clocks | FULL FPS | SPLIT-front FPS | meets 10/30 FPS? | FULL power W | SPLIT power W | FULL J/frame |
SPLIT J/frame | crossover?`

## Honest expected outcomes (state whichever the data shows)
- Our model is light (**10.16 GMACs**); Jetson Orin does tens of TOPS. So full-local **may still hit real-time** on
  the Jetson GPU → **possibly no GPU crossover at 1×**. If so, that is a legitimate finding: for GPU-equipped
  vehicles this light model does not *need* split on compute grounds; split's compute motivation is then specifically
  **(a) CPU-only / weak-accelerator vehicles (E6 CPU arm) and (b) heavier models (fig6)**. Report it plainly.
- The energy/frame ratio should still favor split (≈ the E2 direction) even if both meet real-time — worth stating
  as a battery/thermal point, honestly scoped.
- Whatever happens, the CPU-arm crossover (E6) and the model-scaling extrapolation (fig6) stand independently.

## Deliverables
`results/E7_jetson.md` + raw CSV + a plot mirroring fig1/fig8 (FPS vs power mode, FULL vs SPLIT, with 10/30 FPS
lines). Update `presentation/PRESENTATION.md` §"Does a GPU change it?" and fig8's caption to cite the measured Jetson
result instead of the extrapolation.

## Review rubric (for the sign-off pass)
- Real Jetson, real `nvpmodel` power modes (not a clock hack); FRONT+BACK==FULL verified on-device.
- Power is measured from Jetson rails (tegrastats), not derived; energy/frame reported.
- Crossover claim (if any) tied to a specific power mode + deadline; if none, say so plainly.
- No absolute-watt claims transferred from the RTX 5090 — Jetson numbers are the transferable ones.
