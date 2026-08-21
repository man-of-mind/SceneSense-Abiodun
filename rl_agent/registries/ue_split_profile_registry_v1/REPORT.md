# UE split-profile registry v1 — UE-A1

**Status:** STATIC BINDINGS VERIFIED; UE-A2 WIRE SMOKE REQUIRED

This registry contains all 72 measured actions across 4 model
families, three per-channel quantizers, and six rank-drop fractions. No action
was filtered using perception quality, payload size, or a preferred knob.

## What UE-A1 proves

- every checkpoint and trial summary is present and hash-bound;
- every checkpoint reconstructs with an exact strict state-dictionary match;
- integrated AE weights come from the selected checkpoint; external AE
  override is forbidden;
- model input, expected low/high feature schema, quantizer, q semantics,
  zstd-3 codec, evidence-compatible decoder settings, and distinct host/edge
  checkpoint paths are explicit for every profile;
- the edge profile argument vector is declared, but the current OAI launcher
  does not yet propagate all decoder overrides; and
- all rows remain `REGISTERED_PENDING_SMOKE`.

## Known current-runtime gaps

1. The feature payload does not contain profile/checkpoint/schema/codec
   identity, so the edge cannot reject a mismatched launch. Fixed-profile
   characterization must resolve both sides from the same registry row.
2. Live decoder defaults differ from the retained evidence. Every launch must
   override them with score=0.2,
   NMS=2,
   top-k=120, and published-object
   cap=120.
3. Expected feature shapes are statically registered but remain unobserved on
   the live wire until UE-A2.
4. The current OAI launcher still needs UE-A2 integration for the registered
   edge decoder overrides before any profile is technically valid.

No CARLA run, OAI run, model inference, or policy training was performed.
