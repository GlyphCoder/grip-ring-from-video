import argparse
import json
import os

import cv2
import numpy as np

INNER_D_MM = 56.0
OUTER_D_MM = 86.0

H_LO, H_HI, S_LO, V_LO = 8, 28, 120, 90


def ring_mask(frame_bgr):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((h >= H_LO) & (h <= H_HI) & (s >= S_LO) & (v >= V_LO)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return m


def measure(frame_bgr):
    m = ring_mask(frame_bgr)
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None or len(contours) == 0:
        return None
    hier = hier[0]
    outer_ids = [i for i in range(len(contours)) if hier[i][3] == -1]
    ring = max(outer_ids, key=lambda i: cv2.contourArea(contours[i]))
    if cv2.contourArea(contours[ring]) < 20000:
        return None
    child = hier[ring][2]
    best, best_a = None, 0.0
    while child != -1:
        a = cv2.contourArea(contours[child])
        if a > best_a:
            best, best_a = child, a
        child = hier[child][0]
    if best is None or best_a < 200 or len(contours[best]) < 6:
        return None
    (cx, cy), (d1, d2), ang = cv2.fitEllipse(contours[best])
    minor, major = min(d1, d2), max(d1, d2)
    minor_ang = ang if d1 <= d2 else ang + 90.0
    x, y, w, h = cv2.boundingRect(contours[ring])
    return dict(cx=cx, cy=cy, minor_px=minor, major_px=major,
                minor_axis_deg=minor_ang % 180.0, outer_w_px=float(w),
                outer_h_px=float(h), hole_area_px=best_a)


def track_video(path, fps, out_csv, debug_dir=None):
    cap = cv2.VideoCapture(path)
    rows = []
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        r = measure(fr)
        if r is None:
            rows.append(dict(frame=n, valid=0))
        else:
            r.update(frame=n, valid=1)
            rows.append(r)
            if debug_dir and n % 60 == 0:
                os.makedirs(debug_dir, exist_ok=True)
                vis = fr.copy()
                cv2.ellipse(vis, ((r["cx"], r["cy"]), (r["minor_px"], r["major_px"]),
                                  r["minor_axis_deg"]), (0, 0, 255), 3)
                cv2.imwrite(os.path.join(debug_dir, f"f{n:05d}.jpg"),
                            cv2.resize(vis, None, fx=0.4, fy=0.4))
        n += 1
    cap.release()

    valid = np.array([r["valid"] for r in rows], bool)
    minor = np.array([r.get("minor_px", np.nan) for r in rows])
    major = np.array([r.get("major_px", np.nan) for r in rows])
    outer_w = np.array([r.get("outer_w_px", np.nan) for r in rows])
    area = np.array([r.get("hole_area_px", np.nan) for r in rows])

    from scipy.ndimage import median_filter
    a_med = median_filter(np.nan_to_num(area), size=7, mode="nearest")
    glitch = valid & (np.abs(area - a_med) > 0.25 * a_med)
    valid = valid & ~glitch
    for arr in (minor, major, outer_w):
        arr[~valid] = np.nan

    top = np.nanpercentile(minor, 95)
    rest = valid & (minor >= 0.97 * top)
    px_per_mm = np.nanmedian(outer_w[rest]) / OUTER_D_MM
    hole_rest_mm = np.nanmedian(0.5 * (minor[rest] + major[rest])) / px_per_mm
    outer_check = np.nanmedian(outer_w[rest]) / px_per_mm
    for i, r in enumerate(rows):
        if not valid[i]:
            r["valid"] = 0

    with open(out_csv, "w") as f:
        f.write("frame,t_s,valid,hole_minor_mm,hole_major_mm,outer_w_mm,"
                "minor_axis_deg,cx_px,cy_px\n")
        for i, r in enumerate(rows):
            t = r["frame"] / fps
            if r["valid"]:
                f.write(f"{r['frame']},{t:.6f},1,{r['minor_px']/px_per_mm:.4f},"
                        f"{r['major_px']/px_per_mm:.4f},{r['outer_w_px']/px_per_mm:.4f},"
                        f"{r['minor_axis_deg']:.2f},{r['cx']:.2f},{r['cy']:.2f}\n")
            else:
                f.write(f"{r['frame']},{t:.6f},0,,,,,,\n")
    info = dict(video=os.path.basename(path), fps=fps, frames=n,
                valid_frac=float(valid.mean()), glitches_removed=int(glitch.sum()),
                px_per_mm=float(px_per_mm),
                hole_rest_from_video_mm=float(hole_rest_mm),
                hole_rest_measured_mm=INNER_D_MM,
                rest_ovality=float(np.nanmedian(minor[rest] / major[rest])))
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video")
    ap.add_argument("--fps", type=float)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--trials", default="trials.json")
    ap.add_argument("--out", default="results/tracks")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    jobs = []
    if a.all:
        for t in json.load(open(a.trials))["trials"]:
            jobs.append((t["video"], t["capture_fps"], t["name"]))
    else:
        name = os.path.splitext(os.path.basename(a.video))[0]
        jobs.append((a.video, a.fps, name))

    infos = []
    for video, fps, name in jobs:
        out_csv = os.path.join(a.out, f"{name}.csv")
        dbg = os.path.join(a.out, "debug", name) if a.debug else None
        info = track_video(video, fps, out_csv, dbg)
        infos.append(info)
        print(f"{name}: {info['frames']} frames, {100*info['valid_frac']:.1f}% tracked "
              f"({info['glitches_removed']} glitches removed), {info['px_per_mm']:.2f} px/mm, "
              f"rest hole {info['hole_rest_from_video_mm']:.1f} mm, "
              f"rest ovality {info['rest_ovality']:.3f}")
    json.dump(infos, open(os.path.join(a.out, "tracking_summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
