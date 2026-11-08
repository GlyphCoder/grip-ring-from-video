"""
common.py - shared helpers: load tracks, clean them, find squeeze events,
and run one event through the simulator.
"""
import json
import os

import numpy as np
import pandas as pd

from sim import RingSim

HERE = os.path.dirname(os.path.abspath(__file__))
TRIALS = json.load(open(os.path.join(HERE, "trials.json")))
OBJ = TRIALS["object"]

MASS = OBJ["mass_g"] / 1000.0
R_OUT = OBJ["outer_d_mm"] / 2000.0
HEIGHT = OBJ["height_mm"] / 1000.0
RATED_FORCE_N = OBJ["rated_kg"] * 9.81


# ----------------------------------------------------------------------------
# tracks
# ----------------------------------------------------------------------------
def load_track(name):
    """Tracked hole diameters (mm) with glitches removed, plus rest values."""
    d = pd.read_csv(os.path.join(HERE, "results", "tracks", f"{name}.csv"))
    mn = d.hole_minor_mm.to_numpy(float).copy()
    mj = d.hole_major_mm.to_numpy(float).copy()
    top = np.nanpercentile(mn, 95)
    rest = mn >= 0.97 * top
    rest_minor = float(np.nanmedian(mn[rest]))
    rest_major = float(np.nanmedian(mj[rest]))
    # a squeezed hole only gets wider; a 'major' far below rest means the
    # hole contour split in two (nearly closed) or was misdetected
    bad = (mj < 0.85 * rest_major) | (mn > 1.1 * rest_minor)
    mn[bad] = np.nan
    mj[bad] = np.nan
    return dict(t=d.t_s.to_numpy(float), minor=mn, major=mj,
                rest_minor=rest_minor, rest_major=rest_major,
                n_bad=int(bad.sum()))


def find_events(tr, squeeze_mm=4.0):
    """Split a track into squeeze events. For each: start index, index of
    maximum squeeze, release index (last frame still near the maximum
    squeeze) and recovery timings."""
    t, mn, rest = tr["t"], tr["minor"], tr["rest_minor"]
    filled = pd.Series(mn).ffill().bfill().to_numpy()
    sq = filled < rest - squeeze_mm
    ev = []
    i = 0
    n = len(t)
    while i < n:
        if not sq[i]:
            i += 1
            continue
        j = i
        while j < n and sq[j]:
            j += 1
        # walk back to where the squeeze began (within 1 mm of rest)
        s = i
        while s > 0 and filled[s - 1] < rest - 1.0:
            s -= 1
        seg = filled[i:j]
        k = i + int(np.argmin(seg))
        mmin = filled[k]
        thr = mmin + 0.05 * (rest - mmin)
        r = k
        while r + 1 < n and filled[r + 1] < thr:
            r += 1
        # recovery timing from the release frame
        lvl = lambda p: mmin + p * (rest - mmin)
        def cross(p):
            idx = np.where(filled[r:] >= lvl(p))[0]
            return t[r + idx[0]] - t[r] if len(idx) else np.nan
        e = dict(start=s, peak=k, release=r, end=j,
                 t_start=t[s], t_release=t[r], min_minor=float(mmin),
                 squeeze_mm=float(rest - mmin), hold_s=float(t[r] - t[s]),
                 t10=cross(0.1), t90=cross(0.9), t97=cross(0.97))
        e["rise_10_90_ms"] = 1000 * (e["t90"] - e["t10"])
        # 'snap' = the fingers slipped off faster than the ring recovers, so
        # the recovery is free. Slower releases are limited by the fingers.
        e["snap"] = bool(e["rise_10_90_ms"] < 40.0)
        ev.append(e)
        i = j
    # merge events separated only by a tracking gap (no real recovery between)
    merged = []
    for e in ev:
        if merged and np.nanmax(filled[merged[-1]["release"]:e["start"] + 1]) < rest - 2.0:
            merged[-1].update(end=e["end"])
            continue
        merged.append(e)
    return merged


def all_events(split=None):
    out = []
    for tri in TRIALS["trials"]:
        if split and tri["split"] != split:
            continue
        tr = load_track(tri["name"])
        for k, e in enumerate(find_events(tr)):
            e.update(trial=tri["name"], split=tri["split"], idx=k, fps=tri["capture_fps"])
            out.append(e)
    return out


# ----------------------------------------------------------------------------
# simulator wrapper
# ----------------------------------------------------------------------------
def make_sim(params, r_in):
    s = RingSim(r_in=r_in, r_out=R_OUT, height=HEIGHT, mass=MASS,
                finger_r=params.get("finger_r", 0.008))
    s.set_params(E=params["E"], nu=params["nu"], eta=params["eta"],
                 beta=params["beta"], tau_m=params["tau_m"])
    return s


def static_sweep(params, r_in, min_gap=0.044, n=26):
    """Quasi-static squeeze: finger gap -> hole minor/major (mm) and finger
    force (N). Long-term (relaxed) response."""
    s = make_sim(params, r_in)
    rest_mn, rest_mj = [v * 1e3 for v in s.hole_axes()]
    gaps = np.linspace(2 * R_OUT, min_gap, n)
    rows = []
    for g in gaps:
        mn, mj = s.settle(g, iters=6)
        rows.append((g, mn * 1e3 - rest_mn, mj * 1e3 - rest_mj, s.finger_force()))
    a = np.array(rows)
    return dict(gap=a[:, 0], dminor=a[:, 1], dmajor=a[:, 2], force=a[:, 3])


def gap_from_dminor(sweep, dminor):
    """Invert the static map: finger gap that produces a given change of the
    hole's squeeze-axis diameter (mm). Positive/zero change -> no contact."""
    dm = sweep["dminor"]
    g = sweep["gap"]
    order = np.argsort(dm)
    out = np.interp(dminor, dm[order], g[order])
    return out


def simulate_event(params, tr, ev, sweep, r_in, after_s=0.30, lead_s=0.05,
                   dt_press=2e-3, dt_free=2e-4, record=False):
    """Re-enact one squeeze.

    Press phase (input): the fingers move so that the simulated hole's
      squeeze-axis diameter follows the video (through the static map).
    Release (prediction): at the release frame the fingers are removed and
      the ring recovers on its own ('free'), using only the fitted material.

    Returns sim and observed curves on the video frame times."""
    t, mn = tr["t"], tr["minor"]
    t0 = max(t[0], ev["t_start"] - lead_s)
    tr_ = ev["t_release"]
    t_end = min(t[-1], tr_ + after_s)
    dmn_obs = pd.Series(mn - tr["rest_minor"]).interpolate(limit_direction="both").to_numpy()
    no_contact = 2 * R_OUT

    s = make_sim(params, r_in)
    rest_mn, rest_mj = [v * 1e3 for v in s.hole_axes()]
    frames = np.where((t >= t0) & (t <= t_end))[0]
    out_mn = np.full(len(frames), np.nan)
    out_mj = np.full(len(frames), np.nan)
    xs, gaps = [], []
    gap = None
    fi = 0
    tt = t0
    while fi < len(frames):
        tf = t[frames[fi]]
        # advance to this frame time
        while tt < tf - 1e-9:
            if tt < tr_ - 1e-9:
                dt = min(dt_press, tr_ - tt, tf - tt)
                g = float(gap_from_dminor(sweep, np.interp(tt + dt, t, dmn_obs)))
                gap = None if g >= no_contact - 1e-6 else g
            else:
                dt = min(dt_free, tf - tt)
                gap = None
            s.step(dt, gap)
            tt += dt
        a, b = s.hole_axes()
        if record:
            xs.append(s.x.copy())
            gaps.append(np.nan if gap is None else gap)
        out_mn[fi] = tr["rest_minor"] + (a * 1e3 - rest_mn)
        out_mj[fi] = tr["rest_major"] + (b * 1e3 - rest_mj)
        fi += 1
    res = dict(t=t[frames], frames=frames, t_release=tr_, sim_minor=out_mn,
               sim_major=out_mj, obs_minor=tr["minor"][frames], obs_major=tr["major"][frames])
    if record:
        res.update(x=np.array(xs), gap=np.array(gaps), X=s.X, tris=s.tris,
                   inner=s.inner, outer=s.outer, finger_r=s.finger_r)
    return res


def recovery_error(res, window=(0.0, 0.25)):
    """RMSE (mm) of minor and major over the recovery window after release."""
    m = (res["t"] >= res["t_release"] + window[0]) & (res["t"] <= res["t_release"] + window[1])
    e1 = res["sim_minor"][m] - res["obs_minor"][m]
    e2 = res["sim_major"][m] - res["obs_major"][m]
    e = np.concatenate([e1, e2])
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


def rise_time_ms(t, y, t_release, rest):
    """10-90 % recovery time of the squeeze-axis diameter after release."""
    m = t >= t_release
    t, y = t[m], y[m]
    y0 = y[0]
    def cross(p):
        idx = np.where(y >= y0 + p * (rest - y0))[0]
        return t[idx[0]] if len(idx) else np.nan
    return 1000 * (cross(0.9) - cross(0.1))
