# Cooperative Spatial Map — Architecture, Progress & Plan
*Team briefing · cooperative-perception spatial map · 2026-07*

---

## 0. One-line pitch
**A live, top-down "Google-Maps-for-driving" built on an edge server, where connected cars share only
lightweight *object reports* (not video or heavy tensors) so the map can show every road user around a
car — and, ultimately, warn a car about hazards it physically cannot see (occlusions).**

Think of it as **air-traffic control for an intersection**: each car is a radar station reporting what it
sees; the edge server merges those reports into one shared picture and can tell a "blind" car about a
pedestrian another car can see.

---

## 1. Why we are building this (the problem)
Single-vehicle perception is **blind behind obstacles** — a parked truck hides a crossing child; a building
hides cross-traffic at a junction. Most accidents in intersections/curbsides come from these **occlusions**.

If nearby cars (and roadside sensors) **share what they see**, the blind spots of one car can be filled by
another. The research questions we care about:
1. Can we build a **shared, real-time map** of all detected road users from multiple viewpoints?
2. Can we do it **cheaply on the network** (V2X bandwidth is precious)?
3. Can we **prove a specific car is blind to a specific hazard** (occlusion) and warn it?

---

## 2. The big picture (end-to-end data flow)

```
   ┌────────────── VEHICLE (per car / "UE") ──────────────┐        ┌──────── EDGE SERVER (RSU/MEC) ────────┐
   │  RGB camera + Radar                                   │        │                                        │
   │        │  (7-channel input: 3 RGB + 4 radar)          │        │                                        │
   │        ▼                                              │        │                                        │
   │  [ Perception model — FRONT half ]                    │        │                                        │
   │        │  intermediate features (~1 MB, compressed)   │        │                                        │
   │        └──────── UDP ───────────────────────────────► │  [ Perception model — BACK half ]              │
   │                                                       │        │      │ detections + segmentation      │
   │                                                       │        │      ▼                                 │
   │                                                       │        │  image → WORLD transform (centroid)    │
   │                                                       │        │      │  object list {world x/y, class} │
   │  ◄──────────── (future) targeted alert ───────────────┤◄───────┤      ▼                                 │
   │                                                       │        │  lightweight OBJECT packet (CPM-style) │
   └───────────────────────────────────────────────────────┘        │      │  UDP :39201                     │
                                                                     │      ▼                                 │
                                                                     │  [ MAP SERVER ]  many cars → one map   │
                                                                     │   ingest · place in world · (fuse)     │
                                                                     │   ego-follow ROI · render              │
                                                                     │      │  JSON :35011                    │
                                                                     │      ▼                                 │
                                                                     │  [ Live canvas viewer in browser ]     │
                                                                     └────────────────────────────────────────┘
```

**Key architectural choice:** cars send **object reports** (a few numbers per detected thing), **not** raw
video or the heavy neural-network tensors. This is the ETSI **CPM** (Collective Perception Message) idea and
it's the bandwidth win that makes cooperation practical (see §4).

---

## 3. The building blocks — what each piece does, and why

We walk the pipeline **from the model output on the edge server outward to the map**, because that's where
the interesting map work begins.

### 3.1 The perception model (on the car + edge)
- **What:** an RGB+radar fusion network (MobileNetV3 backbone) that takes a **7-channel** image (3 RGB + 4
  radar) and produces two things:
  - a **segmentation mask** (road / vehicle / pedestrian) — used for the *driver's own dashboard view*, and
  - an **object/localization head** — for each detected object: a **center point**, a **3D position**, a 2D
    box, plus (weak) size and heading.
- **Why radar + camera:** camera gives rich appearance; radar gives **range** (depth). Radar is what lets the
  model place a pedestrian in 3D. Our earlier study showed **more radar points → better pedestrian
  detection, up to ~200k points/sec, then it plateaus** — so we run the **200k-pps model** (accuracy sweet
  spot; costs nothing extra on the wire).

### 3.2 Split inference (front half on car, back half on edge)
- **What:** the network is cut in two. The **front half** runs on the car and produces compact intermediate
  **features**; those are compressed and sent over the network; the **back half** runs on the **edge server**
  and finishes the job (produces the detections + mask).
- **Why:** cars are compute-limited; the edge (RSU/MEC) has GPUs and low latency. Splitting lets a cheap car
  still run a good model. We measured the cost: features are ~1 MB/frame (compressed), front compute ~49 ms,
  back ~8 ms, round-trip ~40 ms on localhost.
- **Important finding:** this feature payload is **the same size regardless of radar point count** (radar is
  rasterized to a fixed-size channel before the split). So higher pps is "free" on the wire.

### 3.3 Image → World transform (turning a detection into a map dot)
This is the heart of "how a camera detection becomes a point on a top-down map."
- The object head **directly regresses each object's 3D position in the sensor's own frame** (how far ahead,
  how far to the side) — it *learned* this from CARLA ground truth, with radar supplying the depth cue.
- Then **one matrix multiply** using the camera's pose (`world = camera_matrix × local_position`) converts
  that into **global world coordinates**. Because the camera rides the moving car, this same step works no
  matter where the car is.
- **Result per object:** `world_x, world_y`, class (vehicle/pedestrian), a confidence score. *(Plus size &
  heading, which we deliberately do not rely on — see §4.)*

### 3.4 The object packet (what a car sends — the "CPM")
A tiny structured message per frame: `stream_id`, the car's **sensor pose + field-of-view**, and a list of
objects (`world x/y`, class, size, score). Hundreds of bytes, not megabytes. Sent over UDP to the map server.

### 3.5 The map server (many cars → one shared picture)
- **Ingest:** listens for object packets from **any number** of cars/poles, each on its own `stream_id`.
- **Common world frame:** every object is already in global coordinates, so the server just drops them onto
  one shared **top-down map** of the town (real road + building geometry drawn as the backdrop).
- **Freshness (TTL):** each car's data expires after a few seconds if it stops reporting, so the map never
  shows stale ghosts.
- **Serves:** a JSON snapshot (`/api/spatial_map/latest`) and a live browser viewer.

### 3.6 The "moving map" (ego-following view)
- The map **crops and re-centers on a chosen car every frame** — a box the size of the model's detection
  range (~40 m), pushed slightly ahead along the car's heading.
- The browser **canvas** polls the JSON ~10×/sec and **interpolates** the car's motion between updates, so the
  world scrolls smoothly under the car — the "Google-Maps-moving-with-the-car" feel.
- **Which view:** it's a **global bird's-eye map in world coordinates** (not a camera view); the *crop*
  follows a chosen ego. Both cars' detections appear in the same shared frame.

### 3.7 The detection representation (how objects are drawn — and what we trust)
- **Position (centroid): trusted (~1.4 m error).** It's the model's prioritized, trained output; the whole
  map rests on it.
- **Size & heading: NOT trusted** — de-prioritized during training. So instead of the model's noisy box, we
  draw **canonical sizes** (a car ≈ 4.6×2.0 m, a pedestrian ≈ 0.8 m) and **snap orientation to the nearest
  road**. Result: clean, realistic, road-aligned boxes.
- **Per-source color + ego markers:** each car's detections get their own color; each car is drawn as its own
  arrow on the map, so you see who is seeing what, and where the cars are relative to the objects.

---

## 4. Key design decisions (the "why", for Q&A)
| Decision | Why |
|---|---|
| **Share objects, not tensors/video** | Bandwidth. Objects are ~hundreds of bytes; tensors are ~1 MB/frame. Object-level makes V2X cooperation practical (the CPM standard does the same). |
| **200k-pps radar model** | Accuracy sweet spot for pedestrians; higher pps doesn't help and costs nothing on the wire. |
| **Trust centroid, replace size/heading** | Training prioritized position; size/yaw are unreliable → canonical sizes + road-snap look correct and stable. |
| **Global world frame on the edge** | Lets any car log in/out at runtime with **zero shared training** — a plug-and-play cooperative map. |
| **Client-side canvas renderer** | Smooth, ~10 fps, GPU-drawn — far lighter than the server rendering an image per frame. |
| **Build incrementally (retire risk)** | Get a plain working multi-car map before adding fusion/occlusion, so each layer is verified before the next. |

---

## 5. What we have working today ✅
1. **End-to-end pipeline live in CARLA**: car drives → model detects (front→edge→back) → objects streamed →
   shared map renders.
2. **Moving ego-map** that smoothly follows the driving car (validated live — "looks like Google Maps").
3. **Clean detection representation**: canonical car/pedestrian footprints, road-aligned, correct centroids.
4. **Two-car support**: a second ego (trailing ~15 m) streams simultaneously; detections colored per car;
   both cars drawn as arrows on the map. *(No fusion yet — each car's detections shown as-is.)*
5. **A first occlusion prototype** running on synthetic ground truth (see §7), reusing a geometry library.
6. Supporting tooling: record/replay of real runs offline, so we can develop without a live simulator.

---

## 6. What is NOT built yet — the planned building blocks
These are the components in the full architecture we have **designed but not yet implemented**, with the
candidate algorithms:

| Block | Job | Candidate approach |
|---|---|---|
| **Data association** | Recognize when two cars are looking at the **same** object | Start: greedy nearest-neighbor. Rigorous: **Hungarian** assignment; **JPDA** if crossings get dense. |
| **Fusion filter** | Merge the matched detections into one better estimate | **Two-view triangulation** (our earlier work: ~1.4 m, beats radar) for position; **Covariance Intersection / EKF** for principled uncertainty. |
| **Temporal tracking** | Keep an object stable frame-to-frame; coast through gaps | Constant-velocity **Kalman filter** per track (also gives velocity → heading, better than model yaw). |
| **Occlusion deduction** | Prove a car is **blind** to an object another car sees | **Field-of-view overlap** + **ray-vs-occluder / visibility grid** (see §7). |
| **Alert feedback loop** | Warn the blind car about the hidden hazard | Transform the hazard into the blind car's frame; send a lightweight "high-alert" trigger. |
| **Robustness studies** | How much GPS/localization error can cooperation tolerate? | Inject controlled pose noise in sim; measure when fusion/occlusion breaks (a clean research contribution). |

**Note on why fusion matters for us specifically:** single-car centroid error is ~1.4 m. When two cars see
the same object, **triangulating** their two views tightens the position — this is where cooperation *earns
its keep*, and it's the next implementation step.

---

## 7. Next focus: the occlusion part (our research novelty)
This is the piece that turns "a nice shared map" into "a safety system," and it's where the novelty is.

**The naive idea (necessary but not sufficient):** if car A sees an object inside car B's field of view but
B doesn't report it, flag a *possible* occlusion. We have this working (reused an existing geometry library),
verified on a synthetic scene (a truck hiding a pedestrian from car B): it correctly flags the hidden
pedestrian. **But on real driving data it over-flags** — it also flags things B simply didn't detect because
they were far away or at the edge of its view. *(Field-of-view membership ≠ proof of occlusion.)*

**The novel step (to build with the team):** add a **geometric occlusion test**. For each object A sees but B
doesn't, cast a ray from **B's sensor to the object**; if it passes through an **occluder** (another car, or
a building footprint) closer than the object → it's a **true occlusion**; otherwise it's just a missed
detection and we drop it. A sharper, radar-native version of this is a **local visibility grid** (free /
occupied / **unknown** cells), inspired by *Dynamic Occupancy Grid* work (radar-centric DOGM, ICRA'24) — the
"unknown" cells are exactly where occlusion lives. We keep the grid **local** to each car and still share
only cheap object lists, so we don't lose the bandwidth advantage.

**How we'll prove it works:** CARLA can tell us the *true* occlusion for each camera (ground truth), so we can
score the method with **precision/recall** — and use a realistic **curbside-accident scenario** (pedestrian
behind a parked vehicle) for the compelling demo.

---

## 8. Suggested talking points / slide skeleton
1. **The problem** — occlusion accidents; single cars are blind (§1).
2. **The vision** — cooperative "ATC for the road," share objects not video (§0, §2).
3. **How a camera detection becomes a map dot** — the 3D-regress + one-matrix-multiply story (§3.3).
4. **Why object-level sharing** — the bandwidth argument, backed by our measurements (§4).
5. **Live demo** — the moving ego-map, two cars, clean boxes (§5). *(show the running viewer / a screenshot)*
6. **What we trust and don't** — centroid yes, size/heading no → canonical + road-snap (§3.7).
7. **The roadmap** — association → fusion (triangulation) → occlusion → alerts (§6).
8. **The novelty & next step** — geometric/visibility-grid occlusion deduction + how we'll validate it (§7).

---

*Companion docs in this folder: `README.md` (how to run), `STAGE3_NOTES.md` (occlusion design detail),
`autonomous_run/GALLERY.md` (figures). Model/accuracy background: `../PPS_STUDY_SUMMARY.md`.*
