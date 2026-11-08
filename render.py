"""
render.py - side-by-side demo video + data for the 3D web viewer.

For each squeeze in the test take (and the training snap):
  left   real video, cropped around the ring
  middle the fitted model, top-down, same scale (fingers drawn while touching)
  bottom hole diameters over time, video (dots) vs model (lines), with cursor

Also exports viewer/data/*.json: the 2D finite-element motion mapped onto a 3D
torus (round tube, the ring's real cross-section), for viewer/index.html.

Writes results/demo.mp4 and viewer/data/
"""
import base64
import json
import os
import subprocess

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import HERE, R_OUT, TRIALS, all_events, load_track

OUT = os.path.join(HERE, "results")
VIEW = os.path.join(HERE, "viewer", "data")
W_PANEL, H_PANEL, H_PLOT = 640, 540, 260
ORANGE = (40, 150, 245)   # BGR
SKIN = (150, 175, 215)


def trial_info(name):
    for t in TRIALS["trials"]:
        if t["name"] == name:
            return t
    raise KeyError(name)


def px_per_mm(name):
    s = json.load(open(os.path.join(OUT, "tracks", "tracking_summary.json")))
    video = os.path.basename(trial_info(name)["video"])
    return [x for x in s if x["video"] == video][0]["px_per_mm"]


def read_frames(path, frames):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    out = []
    for _ in range(len(frames)):
        ok, im = cap.read()
        out.append(im if ok else out[-1])
    cap.release()
    return out


def centre(name, frames):
    import pandas as pd
    d = pd.read_csv(os.path.join(OUT, "tracks", f"{name}.csv"))
    return float(np.nanmedian(d.cx_px.values[frames])), float(np.nanmedian(d.cy_px.values[frames]))


def draw_sim(x, tris, inner, gap, finger_r, scale, size):
    """Top-down drawing of the deformed mesh. x in metres, scale px/m."""
    W, H = size
    img = np.full((H, W, 3), (190, 175, 60), np.uint8)  # teal, like the mat
    P = np.stack([W / 2 + x[:, 0] * scale, H / 2 - x[:, 1] * scale], 1)
    for t in tris:
        cv2.fillConvexPoly(img, np.round(P[t] * 16).astype(np.int32), ORANGE, cv2.LINE_AA, 4)
    for t in tris:
        cv2.polylines(img, [np.round(P[t] * 16).astype(np.int32)], True, (30, 110, 200), 1, cv2.LINE_AA, 4)
    if np.isfinite(gap):
        for sgn in (1, -1):
            c = (W / 2, H / 2 - sgn * (gap / 2 + finger_r) * scale)
            cv2.circle(img, (int(c[0]), int(c[1])), int(finger_r * scale), SKIN, -1, cv2.LINE_AA)
    return img


def curves_image(res, tr, W, H):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0.07, 0.18, 0.9, 0.72])
    t = res["t"] - res["t_release"]
    ax.plot(t, res["obs_minor"], ".", ms=3, color="#1971c2", label="video, along squeeze")
    ax.plot(t, res["sim_minor"], "-", lw=2, color="#1971c2", alpha=0.6, label="model")
    ax.plot(t, res["obs_major"], ".", ms=3, color="#d9480f", label="video, across squeeze")
    ax.plot(t, res["sim_major"], "-", lw=2, color="#d9480f", alpha=0.6, label="model")
    ax.axvline(0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("time from release (s)"); ax.set_ylabel("hole (mm)")
    ax.legend(ncol=4, fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    xlim = ax.get_xlim()
    fig.canvas.draw()
    img = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba())[..., :3], cv2.COLOR_RGB2BGR)
    # pixel x of a data x
    x0 = ax.transData.transform((xlim[0], 0))[0]
    x1 = ax.transData.transform((xlim[1], 0))[0]
    plt.close(fig)
    return img, (xlim, x0, x1)


def label(img, text, y=28, scale=0.6):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    roi = img[y - h - 8:y + 8, 6:18 + w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def event_frames(key, e):
    z = np.load(os.path.join(OUT, f"sim_{key}.npz"))
    res = {k: z[k] for k in z.files}
    tr = load_track(e["trial"])
    info = trial_info(e["trial"])
    frames = res["frames"]
    real = read_frames(os.path.join(HERE, info["video"]), frames)
    ppm = px_per_mm(e["trial"])
    cx, cy = centre(e["trial"], frames)
    half = int(1.25 * R_OUT * 1000 * ppm)  # crop half-size in px
    scale_out = H_PANEL / (2 * half)       # crop -> panel
    sim_scale = ppm * 1000 * scale_out     # px per metre in panel
    plot, (xlim, x0, x1) = curves_image(res, tr, 2 * W_PANEL, H_PLOT)
    kind = "free snap" if e["snap"] else "finger-limited release"
    out, crops = [], []
    for i, f in enumerate(frames):
        im = real[i]
        h, w = im.shape[:2]
        pad = cv2.copyMakeBorder(im, half, half, half, half, cv2.BORDER_REPLICATE)
        crop = pad[int(cy):int(cy) + 2 * half, int(cx):int(cx) + 2 * half]
        crop = cv2.resize(crop, (H_PANEL, H_PANEL))
        crops.append(crop.copy())
        left = np.zeros((H_PANEL, W_PANEL, 3), np.uint8)
        left[:, (W_PANEL - H_PANEL) // 2:(W_PANEL - H_PANEL) // 2 + H_PANEL] = crop
        right = draw_sim(res["x"][i], res["tris"], res["inner"], res["gap"][i],
                         float(res["finger_r"]), sim_scale, (W_PANEL, H_PANEL))
        tt = res["t"][i] - res["t_release"]
        label(left, f"REAL  {info['capture_fps']} fps  ({e['split']}, squeeze #{e['idx']}, {kind})")
        label(right, "MODEL  fitted material, fingers removed at release")
        phase = "pressing (driven by video)" if tt < 0 else "released: model on its own"
        label(right, f"t = {tt*1000:+.0f} ms   {phase}", y=H_PANEL - 16, scale=0.6)
        pl = plot.copy()
        xp = int(x0 + (tt - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0))
        cv2.line(pl, (xp, 10), (xp, H_PLOT - 40), (0, 0, 0), 1)
        out.append(np.vstack([np.hstack([left, right]), pl]))
    return out, res, crops


# ---------------------------------------------------------------------------
# 3D torus export for the web viewer
# ---------------------------------------------------------------------------
def torus(r_in, r_out, n_u=128, n_v=32):
    """Round-tube torus matching the ring (tube diameter = wall width).
    Returns vertices, faces and each vertex's (angle, radius) in the plane."""
    Rc = (r_in + r_out) / 2
    a = (r_out - r_in) / 2
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u, v, indexing="ij")
    rr = Rc + a * np.cos(V)
    P = np.stack([rr * np.cos(U), rr * np.sin(U), a * np.sin(V)], -1).reshape(-1, 3)
    F = []
    for i in range(n_u):
        for j in range(n_v):
            a0 = i * n_v + j
            b0 = ((i + 1) % n_u) * n_v + j
            c0 = ((i + 1) % n_u) * n_v + (j + 1) % n_v
            d0 = i * n_v + (j + 1) % n_v
            F += [[a0, b0, c0], [a0, c0, d0]]
    return P, np.array(F), U.ravel(), rr.ravel()


def grid_weights(theta, r, r_in, r_out, n_theta, n_r):
    """Bilinear weights in the structured annulus mesh (angle x radius), so
    every point of the torus -- including the exact outer and inner edges --
    is tied to the 4 surrounding mesh nodes. (Point-in-triangle lookup fails
    on the edges: the mesh boundary is a polygon, the torus is a circle.)"""
    s = (theta % (2 * np.pi)) / (2 * np.pi) * n_theta
    i0 = np.floor(s).astype(int) % n_theta
    i1 = (i0 + 1) % n_theta
    ft = s - np.floor(s)
    q = np.clip((r - r_in) / (r_out - r_in), 0, 1) * n_r
    j0 = np.minimum(np.floor(q).astype(int), n_r - 1)
    fr = q - j0
    idx = np.stack([j0 * n_theta + i0, j0 * n_theta + i1,
                    (j0 + 1) * n_theta + i0, (j0 + 1) * n_theta + i1], 1)
    w = np.stack([(1 - ft) * (1 - fr), ft * (1 - fr), (1 - ft) * fr, ft * fr], 1)
    return idx, w


def export_viewer(key, e, res, info, metrics):
    os.makedirs(VIEW, exist_ok=True)
    X = res["X"]
    n_theta = len(res["inner"])
    n_r = len(X) // n_theta - 1
    r_in = float(np.linalg.norm(X[res["inner"][0]]))
    r_out = float(np.linalg.norm(X[res["outer"][0]]))
    P, F, th, rr = torus(r_in, r_out)
    # marker dots on the top of the ring, like the pen dots on the real one
    k = np.arange(8)
    mth = 2 * np.pi * (k + 0.5) / 8
    mr = np.full(8, (r_in + r_out) / 2)
    Pm = np.stack([mr * np.cos(mth), mr * np.sin(mth), np.full(8, (r_out - r_in) / 2)], 1)
    allP = np.vstack([P, Pm])
    idx, w = grid_weights(np.concatenate([th, mth]), np.concatenate([rr, mr]),
                          r_in, r_out, n_theta, n_r)
    frames = []
    for x in res["x"]:
        disp = x - X                                   # node displacement (m)
        d = (w[:, :, None] * disp[idx]).sum(1)          # interpolated
        pos = allP.copy()
        pos[:, :2] += d
        q = np.round(pos * 1000 / 0.005).astype(np.int16)   # 5 um steps
        frames.append(base64.b64encode(q.tobytes()).decode())
    d = dict(key=key, trial=e["trial"], squeeze=e["idx"], split=e["split"],
             snap=bool(e["snap"]), fps=info["capture_fps"], video=f"{key}.mp4",
             t=(res["t"] - res["t_release"]).round(5).tolist(),
             faces=F.astype(int).ravel().tolist(), n_vert=len(P), n_markers=8,
             quant_mm=0.005, frames=frames,
             gap_mm=np.where(np.isfinite(res["gap"]), res["gap"] * 1000, -1).round(3).tolist(),
             finger_r_mm=float(res["finger_r"]) * 1000, metrics=metrics,
             obs_minor=np.nan_to_num(res["obs_minor"], nan=-1).round(3).tolist(),
             obs_major=np.nan_to_num(res["obs_major"], nan=-1).round(3).tolist(),
             sim_minor=res["sim_minor"].round(3).tolist(), sim_major=res["sim_major"].round(3).tolist())
    json.dump(d, open(os.path.join(VIEW, f"{key}.json"), "w"))
    return d


def write_mp4(frames, path, fps=30, seekable=False):
    """seekable=True: a keyframe every 5 frames and no B-frames, so the web
    viewer can scrub to an exact frame and stay in sync with the model."""
    h, w = frames[0].shape[:2]
    extra = ["-g", "5", "-bf", "0"] if seekable else []
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                          "-preset", "veryfast", "-crf", "23", *extra, "-pix_fmt", "yuv420p",
                          "-movflags", "+faststart", path],
                         stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    p.wait()


def main():
    import pandas as pd
    import shutil
    metrics = pd.read_csv(os.path.join(OUT, "metrics.csv"))
    fit = json.load(open(os.path.join(OUT, "fit.json")))
    assets = os.path.join(HERE, "viewer", "assets")
    os.makedirs(assets, exist_ok=True)
    for f in ["fit_summary.png", "eval_snaps.png", "eval_rise_times.png", "eval_test.png"]:
        shutil.copy(os.path.join(OUT, f), assets)
    evs = all_events()
    chosen = [e for e in evs if e["split"] == "test"] + [e for e in evs if e["split"] == "train" and e["snap"]]
    all_frames, index = [], []
    for e in chosen:
        key = f"{e['trial']}_{e['idx']}"
        frames, res, crops = event_frames(key, e)
        info = trial_info(e["trial"])
        # title card
        card = np.zeros_like(frames[0])
        kind = "FREE SNAP (fingers slipped off)" if e["snap"] else "FINGER-LIMITED RELEASE"
        for k, s in enumerate([f"{e['split'].upper()} squeeze: {e['trial']} #{e['idx']}", kind,
                               f"played at 30 fps = {info['capture_fps']//30}x slow motion"]):
            cv2.putText(card, s, (40, 200 + 50 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        all_frames += [card] * 45 + frames + [frames[-1]] * 15
        # per-event clip for the web viewer (left panel only, real video)
        os.makedirs(VIEW, exist_ok=True)
        write_mp4(crops, os.path.join(VIEW, f"{key}.mp4"), seekable=True)
        row = metrics[(metrics["trial"] == e["trial"]) & (metrics["squeeze"] == e["idx"])].iloc[0]
        m = dict(press_rmse_mm=float(row.press_widening_rmse_mm),
                 recovery_rmse_mm=float(row.recovery_rmse_mm),
                 rise_obs_ms=float(row.rise_10_90_obs_ms), rise_sim_ms=float(row.rise_10_90_sim_ms),
                 squeeze_mm=float(row.squeeze_mm), hold_s=float(row.hold_s))
        export_viewer(key, e, res, info, m)
        index.append(dict(key=key, split=e["split"], snap=bool(e["snap"]), fps=info["capture_fps"],
                          label=f"{'Test' if e['split']=='test' else 'Train'} squeeze {e['idx']+1}"
                                if e['split'] == 'test' else "Train snap (fit)",
                          kind="free snap" if e["snap"] else "finger-limited", metrics=m))
        print("rendered", key, len(frames), "frames")
    write_mp4(all_frames, os.path.join(OUT, "demo.mp4"))
    p = fit["params"]
    summary = dict(E_MPa=p["E"] / 1e6, eta_over_E_ms=p["eta"] / p["E"] * 1e3, nu=p["nu"],
                   train_snap_rmse_mm=fit["train_recovery_rmse_mm"], events=index)
    json.dump(summary, open(os.path.join(VIEW, "index.json"), "w"), indent=1)
    print("wrote results/demo.mp4 and viewer/data/")


if __name__ == "__main__":
    main()
