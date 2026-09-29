"""Calibrate and run CemaNeige + GR6J (or GR4J; see MODEL in common.py) at
every gauged basin, on a fixed calibration / validation split.

- Simulation starts 2000-01-01; the first 2 years are warm-up (never scored).
- Standard split: calibration 2002-2015, validation 2016-2025, used when the
  gauge has >= 5 observed years in the first and >= 3 in the second.
  Otherwise the observed days after warm-up are split chronologically in
  half (first half calibration). Gauges with < MIN_OBS_YEARS are skipped.
- Objective (OBJECTIVE in common.py): KGE on sqrt(Q) by default, or on Q,
  over the calibration days (differential evolution).

This fixed split is used to evaluate the model and regionalization.
Operational parameters come from annual updating (07_annual_update.py).

Outputs: output/sims/<site>_sim.csv, output/params/<site>.json, output/calibration_summary.csv

Usage:  python 04_calibrate.py [SITE_ID ...]
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")   # one core per worker process

import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import optimize

from common import (CALIB, MIN_OBS_YEARS, MODEL, OBJECTIVE, OUT_DIR, PET_CROP_COEF, PROC_DIR, VALID, WARMUP,
                    ensure_dirs, selected_sites)
from models import BOUNDS, PARAMS, PET_COLUMN, model_inputs, simulate  # noqa: F401

N_WORKERS = min(12, os.cpu_count() or 1)
DE_SETTINGS = dict(maxiter=300, popsize=20, tol=1e-6, polish=True)


# ------------------------------------------------------------------ metrics --
def _valid(obs, sim):
    m = ~np.isnan(obs) & ~np.isnan(sim)
    return obs[m], sim[m]


def kge(obs, sim):
    o, s = _valid(obs, sim)
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def nse(obs, sim):
    o, s = _valid(obs, sim)
    return 1 - np.sum((s - o) ** 2) / np.sum((o - o.mean()) ** 2)


def scores(obs, sim):
    """KGE and its components (r timing, alpha variability, beta volume), NSE,
    NSE on log flows (low-flow skill) and percent bias."""
    o, s = _valid(obs, sim)
    return {"KGE": kge(o, s), "r": float(np.corrcoef(o, s)[0, 1]),
            "alpha": float(s.std() / o.std()), "beta": float(s.mean() / o.mean()),
            "NSE": nse(o, s),
            "logNSE": nse(np.log(o + 0.01), np.log(s + 0.01)),
            "PBIAS": 100 * (s.sum() - o.sum()) / o.sum(), "n_days": int(len(o))}


def split_periods(df):
    """Boolean masks (calibration, validation) and a label for the split used."""
    obs = df.q_obs_mm.notna()
    after_warmup = df.index > WARMUP[1]
    cal = (df.index >= CALIB[0]) & (df.index <= CALIB[1])
    val = (df.index >= VALID[0]) & (df.index <= VALID[1])
    if (obs & cal).sum() >= 5 * 365 and (obs & val).sum() >= 3 * 365:
        return cal, val, f"{CALIB[0][:4]}-{CALIB[1][:4]} / {VALID[0][:4]}-{VALID[1][:4]}"
    days = df.index[obs & after_warmup]
    if len(days) < MIN_OBS_YEARS * 365:
        return None, None, f"only {len(days) / 365:.1f} observed years"
    cut = days[len(days) // 2]
    cal = after_warmup & (df.index < cut)
    val = df.index >= cut
    return cal, val, f"{days[0].year}-{cut.year} / {cut.year}-{days[-1].year} (half split)"


def objective(obs, sim):
    """Calibration score (higher is better)."""
    if OBJECTIVE == "kge_sqrt":
        return kge(np.sqrt(obs), np.sqrt(np.maximum(sim, 0.0)))
    return kge(obs, sim)


def calibrate(inputs, obs, cal_mask, seed=42, x0=None, **de):
    """Maximise the objective over cal_mask days. x0: optional parameter dict
    added to the initial population."""
    idx = np.where(cal_mask & ~np.isnan(obs))[0]
    o = obs[idx]

    def loss(x):
        if not np.all(np.isfinite(x)):
            return 1e3
        q = simulate(inputs, dict(zip(PARAMS, x)))
        k = objective(o, q[idx])
        # A flat hydrograph gives an undefined KGE; penalise it rather than
        # letting NaN reach the gradient-based polishing step.
        return 1.0 - k if np.isfinite(k) else 1e3

    kw = {**DE_SETTINGS, **de}
    if x0 is not None:
        kw["x0"] = [float(np.clip(x0[k], *BOUNDS[k])) for k in PARAMS]
    res = optimize.differential_evolution(
        loss, bounds=[BOUNDS[k] for k in PARAMS], seed=seed, **kw)
    return dict(zip(PARAMS, map(float, res.x)))


def sim_frame(df, st, period):
    """Daily output table: forcing, observed/simulated flow, model states."""
    sim = df[["precip_mm", PET_COLUMN, "temp_c", "q_obs_mm"]].rename(columns={PET_COLUMN: "pet_mm"})
    sim["pet_mm"] = sim["pet_mm"] * PET_CROP_COEF        # PET actually used by the model
    sim["q_sim_mm"] = st["q"]
    sim["snow_swe_mm"] = st["snow"]
    sim["prod_store_mm"] = st["prod"]      # soil moisture (production store)
    sim["rout_store_mm"] = st["rout"]      # routing store
    if "exp" in st:
        sim["exp_store_mm"] = st["exp"]    # GR6J exponential (slow groundwater) store
        sim["aet_mm"] = st["aet"]          # actual evapotranspiration
        sim["exchange_mm"] = st["exchange"]  # net inter-catchment exchange (+ = gain)
    sim["period"] = period
    return sim


# --------------------------------------------------------------------- main --
def run_site(site):
    df = pd.read_csv(PROC_DIR / f"{site}.csv", parse_dates=["date"], index_col="date")
    attrs = json.loads((PROC_DIR / f"{site}_attrs.json").read_text())
    cal, val, split = split_periods(df)
    if cal is None:
        return {"site": site, "name": attrs["name"], "status": f"skipped: {split}"}

    inputs = model_inputs(df, attrs)
    obs = df.q_obs_mm.values
    best = calibrate(inputs, obs, cal)
    st = simulate(inputs, best, return_states=True)
    q = st["q"]
    sim = sim_frame(df, st, np.select([cal, val], ["calibration", "validation"], "warmup"))
    sim.round(4).to_csv(OUT_DIR / "sims" / f"{site}_sim.csv")

    sc_cal, sc_val = scores(obs[cal], q[cal]), scores(obs[val], q[val])
    (OUT_DIR / "params" / f"{site}.json").write_text(json.dumps(
        {"model": MODEL, "objective": OBJECTIVE, "params": best, "split": split, "calibration": sc_cal, "validation": sc_val},
        indent=2))
    row = {"site": site, "name": attrs["name"], "status": "ok", "model": MODEL,
           "objective": OBJECTIVE,
           "split": split, **best}
    row.update({f"{k}_cal": v for k, v in sc_cal.items()})
    row.update({f"{k}_val": v for k, v in sc_val.items()})
    print(f"[{site}] KGE cal={sc_cal['KGE']:.2f} val={sc_val['KGE']:.2f} "
          f"PBIAS val={sc_val['PBIAS']:+.0f}%  {attrs['name'][:40]}", flush=True)
    return row


def main():
    ensure_dirs()
    for d in ("sims", "params"):
        (OUT_DIR / d).mkdir(exist_ok=True)
    sites = [s for s in selected_sites(sys.argv) if (PROC_DIR / f"{s}.csv").exists()]
    print(f"Calibrating CemaNeige-{MODEL.upper()} for {len(sites)} basins on {N_WORKERS} workers ...",
          flush=True)
    with Pool(N_WORKERS) as pool:
        rows = pool.map(run_site, sites, chunksize=1)

    summary = pd.DataFrame(rows).set_index("site")
    path = OUT_DIR / "calibration_summary.csv"
    if path.exists() and len(sys.argv) > 1:   # merge partial re-runs
        old = pd.read_csv(path, dtype={"site": str}).set_index("site")
        summary = pd.concat([old.drop(summary.index, errors="ignore"), summary])
    summary.round(4).to_csv(path)
    ok = summary[summary.status == "ok"]
    print(f"\n{len(ok)} calibrated, {len(summary) - len(ok)} skipped. "
          f"Median KGE cal {ok.KGE_cal.median():.2f}, val {ok.KGE_val.median():.2f}")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
