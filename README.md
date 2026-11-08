# Learning a deformable object from video: a rubber grip ring

**Object:** Decathlon hand-grip ring, "25 kg", 75 g, 86 mm outer diameter, 21 mm tall.
**Interaction:** the ring lies flat on a table; I squeeze it between thumb and finger across a diameter, hold, then let go. Filmed top-down in iPhone slow motion (120 and 240 fps).

**Result in one line:** a 2D finite-element model of the ring, with stiffness, damping and squeeze shape informed by the video, predicts a held-out squeeze to within **0.2–0.4 mm** during the press and **1.3 mm** during the free recovery after release. It also shows clearly which material properties the video can and cannot determine.

![demo frame](results/demo_frame.jpg)

- `results/demo.mp4`: side-by-side real vs simulated, plus live curves (test take and the training snap)
- `viewer/index.html`: interactive 3D viewer (real clip synced with the 3D model, orbit, scrub)

---

## Viewing the results (no Python needed)

Everything is precomputed. The viewer is a static page:

```bash
cd viewer && python -m http.server 8000      # then open http://localhost:8000
```

It must be served (opening `index.html` directly as a file blocks the data loading). It loads three.js from a CDN, so it needs internet. To share a link, enable **GitHub Pages** on the repo (Settings → Pages → deploy from branch) and open `https://<user>.github.io/<repo>/viewer/`. `results/demo.mp4` is the same comparison as a plain video.

## Reproducing everything from the videos

Only needed if you change the videos or the model:

```bash
pip install -r requirements.txt          # numpy scipy opencv-python numba matplotlib pandas tabulate
python track.py --all     # ~1 min:  video -> hole diameters per frame   (results/tracks/*.csv)
python fit.py             # ~5 min:  fit the material                     (results/fit.json, fit_summary.png)
python evaluate.py        # ~1 min:  simulate every squeeze, train + test (results/metrics.md, eval_*.png)
python render.py          # ~30 s:   demo video + 3D viewer data          (results/demo.mp4, viewer/data/)
```

Input videos are in `data/`, and trial metadata (fps, train/test split, ring measurements) is in `trials.json`.

---

## Pipeline

### 1. Observation (`track.py`)
The orange ring is segmented by colour on the teal mat (HSV threshold, morphological close to fill the marker dots). An ellipse is fitted to the **hole** in every frame. The hole is the cleanest signal because the fingers press on the outside and never cover it. Two numbers come out per frame:

- **minor axis**: hole diameter *along* the squeeze
- **major axis**: hole diameter *across* the squeeze

100% of frames are tracked. 47 frames where the hole pinched almost shut and its contour split are discarded automatically (hard presses only).

**Scale.** Pixels → mm uses the ring's outer left-right width, which the fingers never touch, set to 86 mm. This gives 7.2–7.7 px/mm per video.

**A measurement conflict I resolved from the data.** My ruler measurement of the inner diameter was 56 mm, but the video gives **45 mm** in every take. The 45 mm value is the right one:
1. A ring with a round 21 mm tube (matching the 21 mm height) and an 86 mm outside has a ~44 mm hole.
2. With a 45 mm hole, the ring's volume (66.8 cm³) and mass (75 g) give a density of **1.12 g/cm³**, typical for rubber. With a 56 mm hole it would be 1.90 g/cm³, which is implausible. I most likely measured the 56 mm across the rounded inner edge, not the narrowest point.

**Slow-motion time base.** The files are 30 fps containers. Time is `frame / capture_fps`. As a check, the two free snaps (one at 240 fps, one at 120 fps, different takes) have the same 10–90% recovery time (17–21 ms), so the time base is consistent.

### 2. What the observations contain (and don't)
Looking at the recovery after release separates the squeezes into two kinds:

| | 10–90% recovery | what's happening |
|---|---|---|
| **free snap** (2 of 11 squeezes) | 17–21 ms | finger slipped off quickly; the ring recovers on its own |
| **finger-limited** (9 of 11) | 50–233 ms | fingers stay on the ring while opening (visible in the frames), so the curve is my hand's speed |

So only the snaps can teach damping. Luckily there is one in training (take1 #1, 240 fps) and one in the test take (take5_test #2, 120 fps).

### 3. Model (`sim.py`)
- **Representation:** the ring lies flat and is squeezed in its own plane, so its deformation is planar. I model the top-down footprint (annulus, 45/86 mm) with 384 linear triangles in **plane stress**, with the ring height as thickness. Mass is lumped from the real 75 g. For display, the 2D displacement field is mapped onto a 3D torus with the real round cross-section.
- **Material:** co-rotational linear viscoelastic solid. Co-rotation lets the ring bend through large rotations with small local strain:
  `σ = C(E) ε + C(E_m)(ε − ε_v) + C(η) ε̇`, with `ε̇_v = (ε − ε_v)/τ_m`.
  So: stiffness `E`, Poisson ratio `ν`, a fast viscosity `η` (Kelvin–Voigt) and an optional slow relaxing branch (`E_m = βE`, `τ_m`).
- **Fingers:** two rigid discs (fingertips) on the squeeze axis. Contact is an active set of constraints inside the solve. It is unilateral: a node is released when the finger would have to pull on it. It is also sticky (skin grips rubber); without that, the round ring squirts sideways out of two round frictionless fingertips. A small linear drag stands in for table friction.
- **Solver:** linearised backward Euler (Baraff & Witkin 1998) with a sparse direct solve. The rubber is strongly damped, so an explicit integrator would need ~10⁻⁷ s steps. The implicit one runs at 2×10⁻⁴ s (free motion) and 2×10⁻³ s (slow press), so a whole squeeze simulates in about 3 s.
- **Same measurement on both sides:** the model's hole is measured with the same ellipse fit as the video.

### 4. How each observation informs the model (`fit.py`)
For each squeeze, the **press is driven from the video**: the fingers move so the model's along-squeeze diameter follows the video, via a quasi-static gap ↔ diameter map. **At the release frame the fingers are removed**, and from then on the ring moves only under its fitted material. That makes the recovery a prediction, not a replay.

| observation | informs | result |
|---|---|---|
| **shape while squeezed:** how much the hole widens across the squeeze for a given shrink along it (8 training presses) | Poisson ratio ν, fingertip radius | predicted widening within **0.68 mm RMSE** (squeeze < 30 mm). ν itself is *not* identifiable: every value 0.30–0.49 fits within 0.06 mm (kept 0.49, rubber-like). |
| **free snap-back** (training snap) | viscosity η, relaxing branch β, τ_m | **η/E = 16.5 ms**, β → 0 (the plain viscous term already explains the tail), recovery RMSE 1.14 mm |
| **25 kg product rating** (not from video) | absolute stiffness E | **E = 3.3 MPa**, a normal value for rubber |

**Why stiffness comes from the rating, not the video.** This was the most interesting finding.

1. The press is displacement-driven, and the shape of a linear solid under a prescribed displacement doesn't depend on how stiff it is. So the squeeze phase carries no information about E.
2. The snap-back is **heavily overdamped**. With E ≈ 3 MPa and 75 g, an undamped ring would spring back in about 1 ms, but the real one takes ~20 ms with no overshoot. In that regime the response depends on η/E, not on E.

The identifiability scan (`results/fit_summary.png`, right panel) shows this directly: across **E = 0.55 → 17.5 MPa (32×)**, the best snap RMSE only changes from 1.07 to 1.15 mm, while η/E is sharply pinned. My first fit with E free wandered between 2 and 20 MPa for that reason. So E comes from independent information: I assume "25 kg" means the force to squeeze the hole shut and extrapolate the model's linear force curve to closure. With only video, the honest output is "time constant 16.5 ms; stiffness undetermined, ≥ the no-overshoot bound". (One cell of the scan, E = 8.7 MPa at 16.5 ms, spikes to 2.6 mm. That's a contact-solver hiccup, not physics; its neighbours are all 1.1–1.4 mm.)

### 5. Results on the held-out take (`evaluate.py`)
The test take (`pressedhard_120fps_take5_test`, 3 squeezes) was never used for fitting.

| test squeeze | release | predicted widening during press (RMSE) | recovery RMSE | 10–90% rise, video / model |
|---|---|---|---|---|
| #0 | finger-limited | 0.31 mm | 5.4 mm | 50 / 33 ms |
| #1 | finger-limited | 0.15 mm | 4.3 mm | 50 / 33 ms |
| **#2** | **free snap** | **0.43 mm** | **1.34 mm** | 17 / 33 ms |

- **Shape:** the across-squeeze widening is predicted to a fraction of a millimetre for every test squeeze (peak widening 21.1 mm real vs 20.2 mm model on #2).
- **Free recovery (#2):** RMSE 1.34 mm vs 1.14 mm on the training snap, so it generalises to a fresh squeeze at a different depth and frame rate.
- **Finger-limited (#0, #1):** the model, released instantly, recovers faster than the video, as it should. The fingers were still on the ring. These squeezes test the press-phase shape, not the damping.

Full table for all 11 squeezes: `results/metrics.md`. Plots: `results/eval_test.png`, `results/eval_snaps.png`, `results/eval_rise_times.png`.

---

## Assumptions
- Deformation is planar (squeeze in the ring's plane, ring flat on the table). The 3D torus in the viewer is the 2D field mapped onto the real cross-section, not a 3D simulation.
- Homogeneous, isotropic rubber, a linear material law in a co-rotated frame, and plane stress.
- Fingertips are rigid discs that stick to the ring. Their radius is fitted (weakly) from the squeeze shape, and their width along the ring's height is ignored.
- Release is instantaneous at the detected release frame (last frame within 5% of maximum squeeze).
- E comes from interpreting "25 kg" as the force to close the hole (my assumption about the rating).
- The camera looks straight down. The rest hole is slightly oval in the video (minor/major 0.92–0.94, from slight camera tilt or the ring itself), so I compare *changes* from each take's own rest shape.

## Limitations: what the model cannot capture
1. **Absolute stiffness from video alone** (see above). A kitchen scale under the ring, or a known weight placed on it, would turn the rating assumption into a measurement. That's the first thing I'd add.
2. **Very hard squeezes (> 30 mm, hole → slit).** In the real ring the hole collapses into a long slit (widening up to 54 mm). The model saturates at ~34 mm and its mesh folds (`fit_summary.png`, left, red zone). Causes: plane stress ignores the round tube cross-section flattening, there's no self-contact when the hole closes, and the mesh is coarse (4 elements across the wall).
3. **Shape of the snap's start.** The real snap is S-shaped: ~8 ms of slow start while the fingers slide off, then faster than exponential. A first-order overdamped response starts instantly with a longer tail. The 10–90% rise is 17–21 ms real vs 33 ms model, even though the RMSE is good. Modelling the finger withdrawal, or letting inertia matter (lower E), would help; the scan shows lower E fits slightly better.
4. **Damping is an upper bound / "effective".** Only 2 snaps exist, and even a fast finger slip takes a few ms, so η/E = 16.5 ms may include some hand motion. The same time constant at two frame rates and in two takes is reassuring but not proof.
5. **Viscoelastic memory not identified.** The slow-relaxation branch fits to zero on the training snap. Longer holds might reveal creep, but the long-hold squeezes were all finger-limited.
6. **Poisson ratio not identified** by top-down shape alone. It would need a side view (bulging in height) or thickness measurements.
7. Plane-stress vs the real round tube cross-section; table friction as linear drag; the fingers' third dimension ignored.

## Relation to the two papers
- **3D Gaussian Splatting** gives a photoreal static reconstruction. The natural next step here is to reconstruct the ring as Gaussians and drive them with this fitted displacement field (as PhysGaussian does), so the simulation renders like the real video. Then the fit could compare rendered pixels instead of hand-designed ellipse measurements.
- **Particle–voxel dynamics:** my model is a mesh-based (Lagrangian) solver. Particles plus a background grid, MPM-style, would handle what breaks mine, namely self-contact when the hole closes and very large deformation, at the cost of more parameters to identify from the same limited observations.

---

## Files
```
track.py       video -> hole ellipse per frame (OpenCV)
sim.py         co-rotational viscoelastic FEM, implicit integrator, finger contact (NumPy/SciPy/Numba)
common.py      track cleaning, event detection, video-driven event simulation
fit.py         shape -> nu ; rating -> E ; snap -> eta, beta, tau_m ; identifiability scan
evaluate.py    simulate all squeezes, metrics + plots
render.py      demo video + 3D viewer data
viewer/        three.js viewer (index.html + data/)
trials.json    ring measurements, fps, train/test split
data/          input videos
results/       tracks, fit, metrics, plots, demo.mp4
```

## Credits
- OpenCV (segmentation, ellipse fitting), NumPy, SciPy (sparse solver, Nelder–Mead), Numba (element kernels), Matplotlib, pandas, three.js (viewer), FFmpeg.
- Methods: implicit integration after Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998); co-rotational FEM after Müller & Gross, "Interactive Virtual Materials" (GI 2004).
