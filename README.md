# Learning a rubber grip ring from video

I used a Decathlon "25 kg" hand-grip ring (75 g, 86 mm across, 21 mm tall). It lies flat on a table, I squeeze it across a diameter with my thumb and finger, hold it, and let go. I filmed this from above in iPhone slow motion at 120 and 240 fps.

From the video I fit a simple 2D finite-element model of the ring. On a take the model never saw, it predicts the ring's shape while squeezed to within 0.2–0.4 mm, and its free recovery after release to within 1.3 mm.

**Live demo:** [add Vercel link]

![demo frame](results/demo_frame.jpg)

## Running it

The results are already computed, so to see the demo you only need to serve the viewer folder:

```bash
cd viewer
python3 -m http.server 8000     # open http://localhost:8000
```

It needs internet for three.js. `results/demo.mp4` shows the same thing as a normal video.

To rebuild everything from the raw videos:

```bash
pip install -r requirements.txt
python track.py --all    # video -> hole size per frame        (~1 min)
python fit.py            # fit the material                    (~5 min)
python evaluate.py       # simulate every squeeze, get metrics (~1 min)
python render.py         # demo video + viewer data            (~30 s)
```

Videos are in `data/`; frame rates, the train/test split and ring measurements are in `trials.json`.

## What I did

**Tracking.** I segment the orange ring by colour and fit an ellipse to its hole in every frame. I use the hole because my fingers never cover it. That gives the hole size along the squeeze and across it. Pixels are converted to mm using the ring's 86 mm outer width.

My ruler reading of the inner diameter (56 mm) turned out to be wrong. Every video says 45 mm, which gives a density of 1.12 g/cm³, right for rubber (56 mm would give 1.9).

**The releases aren't all the same.** In 9 of my 11 squeezes my fingers stayed on the ring as it opened, so the recovery shows my hand's speed, not the rubber's. Only 2 are real "snaps" where my finger slipped off and the ring recovered by itself in about 20 ms, one in training and one in the test take. Only those can tell the model about damping.

**The model.** The ring is squeezed in its own plane, so I modelled it in 2D: 384 triangles with the ring's height as thickness and its real 75 g mass. Each triangle is a spring (stiffness E) in parallel with a damper (viscosity η), with its rotation removed before computing strain (co-rotational FEM) so the ring can bend a lot. The fingertips are rigid discs that push but don't pull, and grip the rubber; without the grip the ring squirted out sideways.

The rubber is so heavily damped that explicit time-stepping would need 10⁻⁷ s steps, so I used implicit Euler. A squeeze simulates in about 3 s.

**Prediction, not replay.** While I'm pressing, the model's fingers move so its hole shrinks as much as in the video. When I let go, the fingers are deleted and the ring moves only by its fitted material. The sideways widening during the press is also predicted, never given.

**Fitting.**

- *Shape while squeezed* checks the geometry: 0.68 mm error on training presses. It's insensitive to Poisson's ratio, so I used 0.49, typical for rubber.
- *The free snap* gives the damping time constant: η/E = 16.5 ms. An extra slow relaxation term fitted to zero.
- *Stiffness* can't be pinned down by the video. The press is position-controlled, so the shape doesn't depend on stiffness, and the snap is so overdamped that doubling stiffness and damping together gives the same curve. Over a 32× range of stiffness the error barely changed. So I took E from the rating, assuming "25 kg" is the force to close the hole: E = 3.3 MPa, normal for rubber.

## Results

The test take (3 squeezes) wasn't used for any fitting:

| test squeeze | release | widening error while squeezed | recovery error | recovery time (real / model) |
|---|---|---|---|---|
| 1 | fingers stayed on | 0.31 mm | 5.4 mm | 50 / 33 ms |
| 2 | fingers stayed on | 0.15 mm | 4.3 mm | 50 / 33 ms |
| 3 | free snap | 0.43 mm | 1.34 mm | 17 / 33 ms |

The press shape is under a millimetre every time. The free snap's 1.34 mm is close to the training snap's 1.14 mm, so the model holds up on a new squeeze. On the other two it recovers faster than the real ring, as expected, since my fingers were holding the real one back.

All 11 squeezes are in `results/metrics.md`.

## Assumptions

- Flat (2D) deformation; the 3D ring in the viewer is the 2D result wrapped onto the ring's shape.
- Uniform rubber, linear once rotation is removed.
- Rigid fingertips that grip; release is instant.
- "25 kg" is the force to close the hole.
- Camera looks straight down. The hole is slightly oval at rest, so I compare changes from each take's resting shape.

## Limitations

- **Stiffness isn't measured from the video.** A kitchen scale under the ring would fix this; it's the first thing I'd add.
- **Squeezes over 30 mm break it.** The real hole collapses into a slit (up to 54 mm wide); the model stops near 34 mm. No self-contact, coarse mesh, and 2D can't flatten the tube.
- **The snap's start differs.** The real one starts slowly while my finger slides off, then speeds up; the model starts instantly with a longer tail (10–90% time: 17–21 ms real vs 33 ms).
- **Damping comes from two snaps only,** and may include some hand motion, so treat it as an upper bound.
- **No creep or Poisson's ratio:** long holds were all finger-limited, and ν needs a side view.

## Credits

OpenCV, NumPy, SciPy, Numba, Matplotlib, pandas, three.js and FFmpeg. The implicit solver follows Baraff & Witkin, "Large Steps in Cloth Simulation" (SIGGRAPH 1998), and the co-rotational FEM follows Müller & Gross, "Interactive Virtual Materials" (GI 2004).
