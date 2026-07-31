# Density-adaptive knob selection — plan (control-knob-matrix extension)

**Created:** 2026-07-29. Queued from the uplink-only work. **Home:** the control-knob-matrix / RL-agent side
(NOT the staleness analysis — density affects payload only via compression, so it's a *knob-selection* question).

> Feeds the RL agent: adds **scene density** as a state variable and produces a **density → best-knob** policy.
> Judgment-heavy (in-view accuracy, empty-scene metric, artificial-scene trap) → run on **Opus 4.8**, not Haiku.

## Question
For a given scene density (how many cars/pedestrians are in the ego's view), **which compression-knob combination
(quant × ROI × AE) is Pareto-best** — smallest payload/latency while keeping accuracy for the *in-view* objects? And
how does the best knob shift from a dense scene (e.g. 5 vehicles + 5 pedestrians in view) to an empty scene?

Best-guess hypothesis (yes, the joke is basically right): **empty scene → maximal ROI drop (τ→1.0), tiny payload, no
accuracy loss because there's nothing to keep; dense scene → conservative (low ROI τ, u8 bits, no aggressive AE) to
avoid dropping real objects.** The analysis quantifies where that boundary is.

## Physics recap (carry this correction)
The uplink payload is the **backbone feature tensor — fixed-shape regardless of object count** (same size for 2 cars
or 5). So density does NOT grow the raw tensor. Density changes payload **only through content-adaptive compression**:
- **ROI-drop (strong):** ROI keeps cells with objectness > τ. Clear scene → few cells exceed any τ → small payload
  regardless. Dense scene → many cells kept; raising τ drops more but risks dropping weak/partial objects.
- **Entropy coding (mild):** a clear/uniform feature map compresses better than a cluttered one, even at ROI 0.

So the knob's payload *and* its accuracy cost both depend on density — which is exactly why the best knob is
density-conditioned, and why density belongs in the agent's state.

## Method
1. **Density metric + how to bin a continuous drive (same trick as the road-state analysis).** The car drives the
   route normally; density varies naturally (empty stretches vs crowded intersections). **Label each frame post-hoc
   with a GT-derived attribute** — the in-view GT object count (in_camera_frustum, distance ≤ 40 m; split
   vehicles/pedestrians) — then **group frames by that label**, exactly like the old staleness analysis tagged each
   frame curve/straight/intersection from GT and binned by it. No artificial density control during the drive; the
   count is already in the per-frame GT. Bins e.g. `{0, 1–2, 3–4, 5+}`.
   - **Sample-size caveat:** on the standard 28-vehicle / 35-pedestrian drivable route, the high-density bins (5+ in
     the ego frustum at once) may be rare → too few frames to trust. If so, run a **denser NPC scenario** (raise
     `--npc-vehicles`/`--npc-pedestrians`, keep everything else the corrected-drivable) to populate the high bins, or
     widen the top bin — and report `n` per bin. Do NOT read a policy off a bin with a handful of frames.
   - Same confound as road-state: density correlates with location (crowded = intersections, which are also where
     curves/lights are). Control for object distance/speed where possible and state the confound; don't over-claim a
     pure density effect.
2. **Per (density-bin × profile) measurement:** for each knob profile in the matrix (quant u4/u6/u8 × ROI
   {0,0.3,0.5,+high τ toward 1.0} × AE {none,32,64,128}), record **per-frame**: compressed payload bytes, accuracy
   for the *in-view* objects (recall + loc error, matched to in-view GT within 5 m / 40 m, GT **origin** convention),
   and the derived uplink latency (payload → time). Then group frames by density bin and aggregate.
   - Cheapest path: reuse the offline per-model eval (`sweeps_permodel`, `build_knob_matrix.py`) but emit **per-frame
     rows** (payload, matched, in-view GT count) so you can bin by density post-hoc. Or post-process per-frame
     prediction/GT CSVs from captures with payload logged per profile.
3. **Pareto pick per density bin:** the best profile = min payload (≈ min latency) subject to in-view accuracy ≥
   (clean − tolerance). Produce a **density → best-knob lookup table** + the Pareto frontier per bin.
4. **Add scene density to the agent observation** (alongside object speed): the policy raises ROI/compression as
   density falls. Note the observability caveat: the agent can't see the *current* frame's density before sending —
   it uses a proxy (recent detection count from the map / last frame).

## Expected shape of the result
- **Empty (0 in view):** τ→1.0 / most-aggressive AE → near-zero object payload, no recall loss (nothing to keep) —
  the only risk is spurious detections, so also report false-positive rate (should stay ~0).
- **Sparse (1–2):** moderate ROI / smaller bits still safe.
- **Dense (5+):** low ROI τ, u8, no aggressive AE — largest payload, needed to preserve the many (some weak/partial)
  objects; aggressive compression here drops real cars → recall collapse.

## 🚦 Guardrails
1. **Accuracy is measured on the IN-VIEW objects at that density** (not the whole test set). In an empty scene the
   accuracy metric is degenerate (no objects) → any profile "passes" recall → the real metric there is
   **false-positive rate + payload**; report it so "τ=1.0 is best when empty" is justified, not assumed.
2. **Prefer post-hoc density binning on the realistic drivable route over controlled spawns.** Fixed-ego +
   spawn-N-objects scenes are artificial and previously bit us (the Experiment-3 single-target/dead-behind trap,
   F1~0.35). If a controlled density sweep is used, treat it as a clean-isolation *confirmation* only, and flag the
   artificial-scene accuracy caveat.
3. **ROI is already content-adaptive** — frame the "choice" as picking the threshold τ (and bits/AE) *optimally per
   density*, not as inventing a per-object payload. Do not claim density grows the raw tensor.
4. **GT = actor origin**, not bbox-center (the ~1 m offset bug). Anchor the model floor to the offline knob-matrix
   no-AE u8 ≈ 0.95 m, not any loose-matcher live number.
5. **Uplink-only, loopback** for payload→latency mapping; label it; OAI is a separate radio study.
6. **Do NOT export `PYTHONPATH` for any CARLA client** (Session-A lesson, memory `dont_set_pythonpath_for_carla_client`):
   exporting it shadows `abiodun/` with the stale `neu_collab/` copy → `UDPMessageSocket ... unexpected keyword
   'remote_host'`. Analysis/eval scripts that only `import carla` are fine; a CARLA *client* (front/back/loopback) must
   run WITHOUT the export. If a fresh capture is needed: check the machine is idle first (`/proc/loadavg`, no
   OAI/CARLA hogging), reuse a running CARLA rather than launching a duplicate, and don't kill others' processes.
7. **Validate + demote, don't rescue** (Session-A discipline): gate the accuracy data (origin-GT hard-fail; sane
   floor ~1.1 m at v≈0; per-obs direct-vs-closed-form agreement). If a condition fails its gate, demote it and say so;
   salvage only the parts that don't depend on the broken quantity — do not report a rescued number.

## Reuse / outputs / model
- Reuse: `PERMODEL_KNOB_MATRIX_ZSTD.md`, `build_knob_matrix.py`, `../experiments/ae_integrated_20260710/sweeps_permodel`,
  `evaluate_fusion`, and the in-view GT counting from the staleness/eval GT CSVs. The completed uplink-only staleness
  run (`../staleness/uplink_only_latency_budget/`) is a good reference for the obs-loading + origin-GT + gate pattern
  (floor confirmed ~1.1 m; anchor accuracy to the offline knob-matrix 0.95 m, never a loose-matcher live number).
- Outputs → `rl_agent/density_knob/`: `DENSITY_KNOB_RESULTS.md` (density×profile payload/accuracy tables + best-knob
  lookup + Pareto plots per bin), raw CSVs, and a one-line agent-state/policy note for `AGENT_CONSTRAINTS.md`.
- **Model: Opus 4.8** for the analysis; Haiku-high acceptable only for mechanical plotting afterward.
