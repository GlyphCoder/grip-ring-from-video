import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import HERE, all_events, load_track, recovery_error, rise_time_ms, simulate_event

OUT = os.path.join(HERE, "results")


def main():
    fit = json.load(open(os.path.join(OUT, "fit.json")))
    p, r_in = fit["params"], fit["r_in"]
    sweep = {k: np.array(v) for k, v in fit["sweep"].items()}
    print("model:", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items()})

    rows, results = [], {}
    for e in all_events():
        tr = load_track(e["trial"])
        res = simulate_event(p, tr, e, sweep, r_in, after_s=0.35, record=True)
        key = f"{e['trial']}_{e['idx']}"
        results[key] = (e, res, tr)
        np.savez_compressed(os.path.join(OUT, f"sim_{key}.npz"), **{k: v for k, v in res.items()})

        press = res["t"] < res["t_release"]
        sq = press & (res["obs_minor"] < tr["rest_minor"] - 2)
        d = res["sim_major"][sq] - res["obs_major"][sq]
        d = d[np.isfinite(d)]
        rise_obs = rise_time_ms(res["t"], np.nan_to_num(res["obs_minor"], nan=tr["rest_minor"]),
                                res["t_release"], tr["rest_minor"])
        rise_sim = rise_time_ms(res["t"], res["sim_minor"], res["t_release"], tr["rest_minor"])
        rows.append(dict(
            trial=e["trial"], squeeze=e["idx"], split=e["split"],
            release="snap (free)" if e["snap"] else "finger-limited",
            squeeze_mm=round(e["squeeze_mm"], 1), hold_s=round(e["hold_s"], 2),
            press_widening_rmse_mm=round(float(np.sqrt(np.mean(d ** 2))), 2),
            peak_widening_obs_mm=round(float(np.nanmax(res["obs_major"][press]) - tr["rest_major"]), 1),
            peak_widening_sim_mm=round(float(np.nanmax(res["sim_major"][press]) - tr["rest_major"]), 1),
            recovery_rmse_mm=round(recovery_error(res), 2),
            rise_10_90_obs_ms=round(rise_obs, 1), rise_10_90_sim_ms=round(rise_sim, 1),
        ))
        print(f"{key:40s} {rows[-1]['release']:15s} press RMSE {rows[-1]['press_widening_rmse_mm']:.2f} mm, "
              f"recovery RMSE {rows[-1]['recovery_rmse_mm']:.2f} mm, rise obs/sim "
              f"{rise_obs:.0f}/{rise_sim:.0f} ms", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    with open(os.path.join(OUT, "metrics.md"), "w") as f:
        f.write(df.to_markdown(index=False))

    test = [k for k, (e, _, _) in results.items() if e["split"] == "test"]
    fig, axes = plt.subplots(len(test), 2, figsize=(15, 3.6 * len(test)),
                             gridspec_kw=dict(width_ratios=[2.2, 1]), squeeze=False)
    for row, key in zip(axes, test):
        e, res, tr = results[key]
        t = res["t"] - res["t_release"]
        for ax, lim in zip(row, [None, (-0.03, 0.2)]):
            ax.plot(t, res["obs_minor"], ".", ms=4, color="#1971c2", label="video: along squeeze")
            ax.plot(t, res["sim_minor"], "-", lw=2, color="#1971c2", alpha=0.6, label="model")
            ax.plot(t, res["obs_major"], ".", ms=4, color="#d9480f", label="video: across squeeze")
            ax.plot(t, res["sim_major"], "-", lw=2, color="#d9480f", alpha=0.6, label="model")
            ax.axvline(0, color="k", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            ax.set_ylabel("hole diameter (mm)")
            if lim:
                ax.set_xlim(*lim)
                ax.set_title("recovery (zoom)")
            else:
                kind = "free snap" if e["snap"] else "finger-limited release"
                ax.set_title(f"TEST {e['trial']} squeeze #{e['idx']} ({kind})")
                ax.legend(ncol=2, fontsize=8)
        row[0].set_xlabel("time from release (s)"); row[1].set_xlabel("time from release (s)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eval_test.png"), dpi=120)

    snaps = [k for k, (e, _, _) in results.items() if e["snap"]]
    fig, ax = plt.subplots(1, len(snaps), figsize=(6.5 * len(snaps), 4.2), squeeze=False)
    for a, key in zip(ax[0], snaps):
        e, res, tr = results[key]
        t = (res["t"] - res["t_release"]) * 1e3
        m = (t > -20) & (t < 200)
        a.plot(t[m], res["obs_minor"][m], "o", ms=4, color="#1971c2", label="video, along squeeze")
        a.plot(t[m], res["sim_minor"][m], "-", lw=2, color="#1971c2", label="model")
        a.plot(t[m], res["obs_major"][m], "o", ms=4, color="#d9480f", label="video, across squeeze")
        a.plot(t[m], res["sim_major"][m], "-", lw=2, color="#d9480f", label="model")
        a.set_title(f"{e['split'].upper()}: {e['trial']} #{e['idx']}\n"
                    f"recovery RMSE {recovery_error(res):.2f} mm")
        a.set_xlabel("ms after release"); a.set_ylabel("hole diameter (mm)")
        a.grid(alpha=0.3); a.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eval_snaps.png"), dpi=120)

    fig, a = plt.subplots(figsize=(11, 4))
    x = np.arange(len(df))
    a.bar(x - 0.2, df.rise_10_90_obs_ms, 0.4, label="video", color="#868e96")
    a.bar(x + 0.2, df.rise_10_90_sim_ms, 0.4, label="model (fingers removed)", color="#d9480f")
    a.set_xticks(x)
    a.set_xticklabels([f"{r.trial.split('_')[0][7:]} {r.trial.split('_')[2]}\n#{r.squeeze} {r.split}"
                       + ("\nSNAP" if r.release.startswith("snap") else "")
                       for r in df.itertuples()], fontsize=7)
    a.set_ylabel("10-90% recovery time (ms)")
    a.set_title("Recovery speed: snaps match the model; slow releases are limited by the fingers")
    a.legend(); a.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "eval_rise_times.png"), dpi=120)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
