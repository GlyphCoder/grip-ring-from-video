import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from common import (HERE, RATED_FORCE_N, TRIALS, all_events, load_track,
                    recovery_error, simulate_event, static_sweep)

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)


def mean_rest_hole_radius():
    s = json.load(open(os.path.join(OUT, "tracks", "tracking_summary.json")))
    return float(np.mean([x["hole_rest_from_video_mm"] for x in s])) / 2000.0


def press_samples(events):
    xs, ys = [], []
    for e in events:
        tr = load_track(e["trial"])
        sl = slice(e["start"], e["release"] + 1)
        dmn = tr["minor"][sl] - tr["rest_minor"]
        dmj = tr["major"][sl] - tr["rest_major"]
        ok = np.isfinite(dmn) & np.isfinite(dmj) & (dmn < -2.0)
        xs.append(dmn[ok]); ys.append(dmj[ok])
    return np.concatenate(xs), np.concatenate(ys)


VALID_SQUEEZE_MM = 30.0


def fit_static(r_in, events):
    x, y = press_samples(events)
    v = x > -VALID_SQUEEZE_MM
    base = dict(E=3e6, eta=1e4, beta=0.0, tau_m=0.1)
    grid = []
    t0 = time.time()
    for nu in [0.30, 0.35, 0.40, 0.45, 0.49]:
        for rf in [0.005, 0.008, 0.011]:
            sw = static_sweep(dict(base, nu=nu, finger_r=rf), r_in, min_gap=0.034, n=28)
            o = np.argsort(sw["dminor"])
            pred = np.interp(x, sw["dminor"][o], sw["dmajor"][o])
            rmse = float(np.sqrt(np.mean((pred[v] - y[v]) ** 2)))
            rmse_hard = float(np.sqrt(np.mean((pred[~v] - y[~v]) ** 2)))
            grid.append(dict(nu=nu, finger_r=rf, rmse_mm=rmse, rmse_beyond_valid_mm=rmse_hard))
            print(f"  nu={nu:.2f} r_f={rf*1e3:.0f} mm  ->  widening RMSE {rmse:.2f} mm "
                  f"(squeeze < {VALID_SQUEEZE_MM:.0f} mm), {rmse_hard:.1f} mm beyond")
    best = min(grid, key=lambda g: g["rmse_mm"])
    print(f"  best: nu={best['nu']} r_f={best['finger_r']*1e3:.0f} mm "
          f"({time.time()-t0:.0f}s)")
    return best, grid, (x, y)


def calibrate_E(r_in, nu, rf):
    E_ref = 1e6
    sw = static_sweep(dict(E=E_ref, nu=nu, eta=1e4, beta=0.0, tau_m=0.1, finger_r=rf),
                      r_in, min_gap=0.030, n=30)
    rest_hole = 2 * r_in * 1e3
    closed = -rest_hole
    x, f = sw["dminor"], sw["force"]
    k = (x < -5) & (x > -30)
    p = np.polyfit(x[k], f[k], 1)
    F_closed = float(np.polyval(p, closed))
    E = E_ref * RATED_FORCE_N / F_closed
    print(f"  model force to close the hole at E=1 MPa: {F_closed:.0f} N  ->  "
          f"E = {E/1e6:.2f} MPa for {RATED_FORCE_N:.0f} N (25 kg)")
    return E, sw


def fit_dynamics(r_in, static_params, snaps, sweeps):
    tracks = {e["trial"]: load_track(e["trial"]) for e in snaps}
    hist = []
    E = static_params["E"]

    def unpack(z):
        return dict(static_params, eta=float(E * np.exp(z[0])),
                    beta=float(np.clip(z[1], 0.0, 3.0)),
                    tau_m=float(np.exp(np.clip(z[2], np.log(0.01), np.log(3.0)))))

    def loss(z):
        p = unpack(z)
        err = [recovery_error(simulate_event(p, tracks[e["trial"]], e, sweeps, r_in))
               for e in snaps]
        L = float(np.mean(err))
        hist.append(dict(p, rmse_mm=L))
        print(f"    eta/E={p['eta']/E*1e3:.1f}ms beta={p['beta']:.2f} "
              f"tau_m={p['tau_m']*1e3:.0f}ms -> {L:.3f} mm", flush=True)
        return L

    z0 = np.array([np.log(0.015), 0.15, np.log(0.08)])
    simplex = [z0, z0 + [0.5, 0, 0], z0 + [0, 0.3, 0], z0 + [0, 0, 1.0]]
    r = minimize(loss, z0, method="Nelder-Mead",
                 options=dict(maxfev=60, xatol=0.02, fatol=0.003, initial_simplex=simplex))
    return unpack(r.x), float(r.fun), hist


def scan_identifiability(r_in, best, snaps, sweeps):
    tracks = {e["trial"]: load_track(e["trial"]) for e in snaps}
    Es = best["E"] * np.array([1 / 6, 1 / 3, 2 / 3, 4 / 3, 8 / 3, 16 / 3])
    taus = best["eta"] / best["E"] * np.array([0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.2])
    L = np.zeros((len(Es), len(taus)))
    for i, E in enumerate(Es):
        for j, tau in enumerate(taus):
            p = dict(best, E=E, eta=E * tau)
            L[i, j] = np.mean([recovery_error(simulate_event(p, tracks[e["trial"]], e, sweeps, r_in))
                               for e in snaps])
        print(f"    E={E/1e6:.2f} MPa: " + " ".join(f"{v:.2f}" for v in L[i]), flush=True)
    return Es, taus, L


def main():
    t0 = time.time()
    r_in = mean_rest_hole_radius()
    train = all_events("train")
    snaps = [e for e in train if e["snap"]]
    print(f"rest hole radius {r_in*1e3:.1f} mm; {len(train)} training squeezes, "
          f"{len(snaps)} free snap(s): " + ", ".join(f"{e['trial']} #{e['idx']}" for e in snaps))

    print("\n[1] squeeze shape -> Poisson ratio, fingertip radius")
    best_s, grid, (x, y) = fit_static(r_in, train)

    print("\n[2] stiffness from the 25 kg rating")
    E, sw_force = calibrate_E(r_in, best_s["nu"], best_s["finger_r"])
    static_params = dict(E=E, nu=best_s["nu"], finger_r=best_s["finger_r"])

    sweep = static_sweep(dict(static_params, eta=1e4, beta=0.0, tau_m=0.1), r_in,
                         min_gap=0.030, n=34)

    print("\n[3] snap-back -> viscosity and relaxation")
    best, best_err, hist = fit_dynamics(r_in, static_params, snaps, sweep)
    print(f"  fitted: eta={best['eta']:.3g} Pa*s (eta/E = {best['eta']/best['E']*1e3:.1f} ms), "
          f"beta={best['beta']:.2f}, tau_m={best['tau_m']*1e3:.0f} ms, RMSE {best_err:.2f} mm")

    print("\n[4] identifiability scan over (E, eta/E)")
    Es, taus, L = scan_identifiability(r_in, best, snaps, sweep)

    res = dict(r_in=r_in, params=best, train_recovery_rmse_mm=best_err,
               static_grid=grid, static_best=best_s,
               E_from_rating=E, rated_force_N=RATED_FORCE_N,
               snaps=[f"{e['trial']}#{e['idx']}" for e in snaps],
               ident=dict(E=Es.tolist(), tau=taus.tolist(), rmse=L.tolist()),
               sweep={k: v.tolist() for k, v in sweep.items()},
               force_sweep={k: v.tolist() for k, v in sw_force.items()},
               history=hist)
    json.dump(res, open(os.path.join(OUT, "fit.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    o = np.argsort(sweep["dminor"])
    ax[0].plot(-x, y, ".", ms=2, alpha=0.4, color="#888", label="video (train presses)")
    ax[0].plot(-sweep["dminor"][o], sweep["dmajor"][o], "-", lw=2.5, color="#d9480f",
               label=f"model, nu={best_s['nu']}, r_f={best_s['finger_r']*1e3:.0f} mm")
    ax[0].set_xlabel("hole shrinks along squeeze (mm)")
    ax[0].set_ylabel("hole widens across squeeze (mm)")
    ax[0].axvspan(VALID_SQUEEZE_MM, ax[0].get_xlim()[1], color="#ffe3e3", alpha=0.6, zorder=0)
    ax[0].text(VALID_SQUEEZE_MM + 0.5, 3, "model breaks down\n(hole -> slit)", fontsize=8, color="#c92a2a")
    ax[0].set_title("Squeeze shape: predicted widening"); ax[0].legend(loc="upper left"); ax[0].grid(alpha=0.3)

    fs = sw_force
    ax[1].plot(-np.array(fs["dminor"]), np.array(fs["force"]) * E / 1e6, "-o", ms=3, color="#1971c2")
    ax[1].axhline(RATED_FORCE_N, ls="--", color="k", lw=1)
    ax[1].text(1, RATED_FORCE_N * 1.02, "25 kg rating (assumed = force to close the hole)", fontsize=8)
    ax[1].set_xlabel("hole shrinks along squeeze (mm)"); ax[1].set_ylabel("finger force (N)")
    ax[1].set_title(f"Force (E = {E/1e6:.2f} MPa from rating)"); ax[1].grid(alpha=0.3)

    im = ax[2].imshow(L, origin="lower", aspect="auto", cmap="viridis_r",
                      extent=[-0.5, len(taus) - 0.5, -0.5, len(Es) - 0.5])
    ax[2].set_xticks(range(len(taus))); ax[2].set_xticklabels([f"{t*1e3:.1f}" for t in taus])
    ax[2].set_yticks(range(len(Es))); ax[2].set_yticklabels([f"{e/1e6:.2f}" for e in Es])
    ax[2].set_xlabel("eta / E  (ms)"); ax[2].set_ylabel("E (MPa)")
    ax[2].set_title("Snap-back RMSE (mm): eta/E is pinned, E is not")
    plt.colorbar(im, ax=ax[2])
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fit_summary.png"), dpi=130)
    print(f"\ndone in {time.time()-t0:.0f}s -> results/fit.json, results/fit_summary.png")


if __name__ == "__main__":
    main()
