"""Rolling-origin test: does recalibrating each year improve next-year prediction?

For every target year Y in 2016-2025, parameters are calibrated using only
data before Y, then used to predict Y (never seen in calibration):

  prev1      calibrate on year Y-1 only           (annual recalibration)
  prev5      calibrate on years Y-5 .. Y-1        (moving window)
  expanding  calibrate on 2002 .. Y-1             (all past data)
  fixed      calibrate once on 2002-2015          (current approach)

Every simulation starts 2000-01-01 so model stores carry over realistically;
only the scored calibration window differs. All strategies use identical
optimiser settings (cheaper than 04_calibrate.py, so the fixed baseline is
recalibrated here too for a fair comparison).

Outputs: output/experiments/rolling_origin.csv, rolling_origin_params.csv,
         rolling_origin.png

Usage:  python experiments/rolling_origin.py [N_BASINS]
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, PROC_DIR  # noqa: E402

cal = importlib.import_module("04_calibrate")

TARGETS = range(2016, 2026)
STRATEGIES = ["prev1", "prev5", "expanding", "fixed"]
DE = dict(maxiter=100, popsize=15, tol=1e-6, polish=True)
EXP_DIR = OUT_DIR / "experiments"


def window(strategy, year):
    if strategy == "prev1":
        return year - 1, year - 1
    if strategy == "prev5":
        return year - 5, year - 1
    if strategy == "expanding":
        return 2002, year - 1
    return 2002, 2015


def load(site):
    df = pd.read_csv(PROC_DIR / f"{site}.csv", parse_dates=["date"], index_col="date")
    attrs = json.loads((PROC_DIR / f"{site}_attrs.json").read_text())
    return df, attrs


def fit(task):
    """Calibrate one (site, strategy, window); simulation ends at window end."""
    site, strategy, y0, y1 = task
    df, attrs = load(site)
    df = df.loc[: f"{y1}-12-31"]
    inputs = cal.model_inputs(df, attrs)
    obs = df.q_obs_mm.values
    idx = np.where((df.index.year >= y0) & ~np.isnan(obs))[0]

    def loss(x):
        q = cal.simulate(inputs, dict(zip(cal.PARAMS, x)))
        return 1.0 - cal.kge(obs[idx], q[idx])

    res = optimize.differential_evolution(
        loss, bounds=[cal.BOUNDS[k] for k in cal.PARAMS], seed=42, **DE)
    return (site, strategy, y0, y1), dict(zip(cal.PARAMS, map(float, res.x))), 1 - res.fun


def choose_basins(n):
    """Long-record basins, spread evenly across the tile-drainage range."""
    summ = pd.read_csv(OUT_DIR / "calibration_summary.csv", dtype={"site": str}).set_index("site")
    att = pd.read_csv(PROC_DIR / "basin_attributes.csv", dtype={"site": str}).set_index("site")
    ok = []
    for s in summ[summ.status == "ok"].index:
        q = load(s)[0].q_obs_mm.loc["2001":"2025"]
        if q.groupby(q.index.year).count().ge(340).sum() == 25:     # complete 2001-2025
            ok.append(s)
    ok = att.loc[ok].sort_values("tile_frac").index.tolist()
    pick = np.unique(np.linspace(0, len(ok) - 1, min(n, len(ok))).round().astype(int))
    return [ok[i] for i in pick], len(ok)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    sites, n_long = choose_basins(n)
    print(f"{len(sites)} of {n_long} complete-record basins, targets {TARGETS[0]}-{TARGETS[-1]}")

    tasks = {(s, st, *window(st, y)) for s in sites for st in STRATEGIES for y in TARGETS}
    tasks = sorted(tasks)
    print(f"{len(tasks)} calibrations on {cal.N_WORKERS} workers ...", flush=True)
    params = {}
    with Pool(cal.N_WORKERS) as pool:
        for i, (key, p, k) in enumerate(pool.imap_unordered(fit, tasks), 1):
            params[key] = (p, k)
            if i % 25 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)
    pd.DataFrame([{"site": k[0], "strategy": k[1], "y0": k[2], "y1": k[3],
                   "KGE_calib": v[1], **v[0]} for k, v in params.items()]
                 ).to_csv(EXP_DIR / "rolling_origin_params.csv", index=False)

    rows = []
    for s in sites:
        df, attrs = load(s)
        inputs = cal.model_inputs(df, attrs)
        obs = df.q_obs_mm.values
        for st in STRATEGIES:
            for y in TARGETS:
                p, _ = params[(s, st, *window(st, y))]
                q = cal.simulate(inputs, p)
                m = df.index.year == y
                o, sim = obs[m], q[m]
                ok = ~np.isnan(o)
                rows.append({"site": s, "strategy": st, "year": y,
                             "KGE": cal.kge(o, sim),
                             "NSE": cal.nse(o, sim),
                             "vol_err_pct": 100 * (sim[ok].sum() - o[ok].sum()) / o[ok].sum(),
                             **{f"p_{k}": v for k, v in p.items()}})
    res = pd.DataFrame(rows)
    res.to_csv(EXP_DIR / "rolling_origin.csv", index=False)

    print("\nNext-year prediction skill, all basin-years (2016-2025):")
    g = res.groupby("strategy")
    table = pd.DataFrame({
        "median KGE": g.KGE.median(), "25th pct KGE": g.KGE.quantile(0.25),
        "share KGE<0": g.KGE.apply(lambda v: (v < 0).mean()),
        "median |vol err| %": g.vol_err_pct.apply(lambda v: v.abs().median()),
    }).loc[STRATEGIES]
    print(table.round(3).to_string())

    wide = res.pivot_table(index=["site", "year"], columns="strategy", values="KGE")
    print("\nHead-to-head vs fixed (share of basin-years where strategy beats fixed):")
    for st in STRATEGIES[:-1]:
        print(f"  {st:9s} {(wide[st] > wide['fixed']).mean():.0%}")

    # Parameter stability: year-to-year change in x1 under each strategy.
    stab = res.groupby(["strategy", "site"]).p_x1.agg(lambda v: v.std() / v.mean())
    print("\nParameter instability: median coefficient of variation of x1 across target years")
    print(stab.groupby("strategy").median().loc[STRATEGIES].round(3).to_string())
    print("\nMedian KGE by target year:")
    print(res.pivot_table(index="strategy", columns="year", values="KGE", aggfunc="median")
          .loc[STRATEGIES].round(2).to_string())

    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    COLORS = {"prev1": "#eb6834", "prev5": "#1baf7a", "expanding": "#2a78d6", "fixed": INK}
    LABELS = {"prev1": "Previous year only", "prev5": "Previous 5 years",
              "expanding": "All past years", "fixed": "Fixed 2002-2015"}
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "axes.edgecolor": "#8a8984", "xtick.color": INK2, "ytick.color": INK2,
                         "axes.labelcolor": INK2, "font.size": 9.5, "axes.titleweight": "bold",
                         "axes.titlelocation": "left", "legend.frameon": False})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    rng = np.random.default_rng(0)
    for i, st in enumerate(STRATEGIES):
        v = res.loc[res.strategy == st, "KGE"].clip(lower=-1)
        a1.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=9, alpha=0.35, color=COLORS[st], lw=0)
        med = res.loc[res.strategy == st, "KGE"].median()
        a1.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=2.2)
        a1.text(i + 0.35, med, f"{med:.2f}", va="center", fontsize=9)
    a1.set_xticks(range(len(STRATEGIES)), [LABELS[s].replace(" ", "\n", 1) for s in STRATEGIES])
    a1.set_ylabel("KGE of next-year prediction")
    a1.set_ylim(-1.05, 1)
    a1.set_title("All basin-years, 2016-2025 (bar = median; < −1 clipped)")
    med = res.pivot_table(index="year", columns="strategy", values="KGE", aggfunc="median")
    for st in STRATEGIES:
        a2.plot(med.index, med[st], marker="o", ms=5, lw=2, color=COLORS[st], label=LABELS[st])
    a2.set_xticks(list(TARGETS))
    a2.set_ylabel("Median KGE across basins")
    a2.set_title("Skill by target year")
    a2.legend(ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(EXP_DIR / "rolling_origin.png", dpi=150, bbox_inches="tight")
    print("wrote", EXP_DIR / "rolling_origin.png")


if __name__ == "__main__":
    main()
