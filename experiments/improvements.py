"""Controlled test of three candidate improvements (GR6J, KGE on sqrt Q,
fixed split 2002-2015 / 2016-2025, identical optimiser settings):

  base           as configured
  shift-1        precipitation moved one day earlier  } day-boundary
  shift+1        precipitation moved one day later    } alignment check
  kc0.78         PET x 0.78 (crop coefficient used by iowa-nitrates)
  kc_cal         PET multiplier calibrated per basin, bounds 0.5-1.2

Outputs: output/experiments/improvements.csv (per basin x variant),
         improvements_summary.csv

Usage:  python experiments/improvements.py
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("IWB_MODEL", "gr6j")

import importlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT_DIR, PROC_DIR  # noqa: E402
import models  # noqa: E402

cal = importlib.import_module("04_calibrate")

VARIANTS = {
    "base": dict(shift=0, kc=1.0),
    "shift-1": dict(shift=-1, kc=1.0),
    "shift+1": dict(shift=1, kc=1.0),
    "kc0.78": dict(shift=0, kc=0.78),
    "kc_cal": dict(shift=0, kc=None),
}
KC_BOUNDS = (0.5, 1.2)
DE = dict(maxiter=150, popsize=15, tol=1e-6, polish=True, seed=42)
EXP_DIR = OUT_DIR / "experiments"


def job(args):
    site, name = args
    v = VARIANTS[name]
    df = pd.read_csv(PROC_DIR / f"{site}.csv", parse_dates=["date"], index_col="date")
    attrs = json.loads((PROC_DIR / f"{site}_attrs.json").read_text())
    c, val, _ = cal.split_periods(df)
    if v["shift"]:
        df["precip_mm"] = df.precip_mm.shift(v["shift"]).fillna(0.0)
    prec, tmean, frac, etp = models.model_inputs(df, attrs)
    if v["shift"]:   # snow partition must follow the shifted precipitation
        frac = np.roll(frac, v["shift"], axis=0)
    obs = df.q_obs_mm.values
    idx = np.where(c & ~np.isnan(obs))[0]
    names = list(models.PARAMS) + (["kc"] if v["kc"] is None else [])
    bounds = [models.BOUNDS[k] for k in models.PARAMS] + ([KC_BOUNDS] if v["kc"] is None else [])

    def run(x):
        p = dict(zip(names, x))
        kc = p.pop("kc", v["kc"])
        return models.simulate((prec, tmean, frac, etp * kc), p, return_states=True), kc

    def loss(x):
        if not np.all(np.isfinite(x)):
            return 1e3
        st, _ = run(x)
        k = cal.objective(obs[idx], st["q"][idx])
        return 1.0 - k if np.isfinite(k) else 1e3

    res = optimize.differential_evolution(loss, bounds, **DE)
    st, kc = run(res.x)
    q = st["q"]
    sv = cal.scores(obs[val], q[val])
    yrs = df.index.year
    ex = st["exchange"][(yrs >= 2002)].sum() / (yrs >= 2002).sum() * 365.25
    return {"site": site, "variant": name, "kc": kc,
            **{f"{k}_val": x for k, x in sv.items()},
            "exchange_mm_yr": ex, **dict(zip(names, res.x))}


def main():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(OUT_DIR / "calibration_summary.csv", dtype={"site": str}).set_index("site")
    sites = list(summ[summ.status == "ok"].index)
    jobs = [(s, v) for s in sites for v in VARIANTS]
    print(f"{len(jobs)} calibrations ({len(sites)} basins x {len(VARIANTS)} variants)", flush=True)
    with Pool(cal.N_WORKERS) as pool:
        res = pool.map(job, jobs, chunksize=1)
    d = pd.DataFrame(res)
    d.to_csv(EXP_DIR / "improvements.csv", index=False)

    rows = []
    base = d[d.variant == "base"].set_index("site")
    for name in VARIANTS:
        x = d[d.variant == name].set_index("site")
        r = {"variant": name, "KGE_val": x.KGE_val.median(), "r": x.r_val.median(),
             "alpha": x.alpha_val.median(), "beta": x.beta_val.median(),
             "logNSE": x.logNSE_val.median(), "PBIAS": x.PBIAS_val.median(),
             "share_KGE>=0.75": (x.KGE_val >= 0.75).mean(),
             "exchange_mm_yr": x.exchange_mm_yr.median()}
        if name != "base":
            diff = (x.KGE_val - base.KGE_val).dropna()
            r["dKGE_vs_base"] = diff.median()
            r["better_share"] = (diff > 0).mean()
            r["wilcoxon_p"] = wilcoxon(diff).pvalue
        rows.append(r)
    out = pd.DataFrame(rows).set_index("variant")
    out.round(3).to_csv(EXP_DIR / "improvements_summary.csv")
    print(out.round(3).to_string())
    kc = d[d.variant == "kc_cal"].kc
    print(f"\nCalibrated Kc: median {kc.median():.2f} (IQR {kc.quantile(.25):.2f}-{kc.quantile(.75):.2f})")


if __name__ == "__main__":
    main()
