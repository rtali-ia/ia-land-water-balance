"""Annual updating: calibrate on all prior history, predict the next year.

For each basin and each target year Y = FIRST_TARGET .. last data year + 1:
  1. calibrate on every observed day from 2002-01-01 to Y-1 (expanding window),
     from scratch each year (seeding with last year's optimum was tested and
     locked in poor early solutions);
  2. simulate from 2000-01-01 with those parameters (stores carry over), and
     keep year Y as the prediction for that year.
Stitching the Y slices gives a fully out-of-sample hindcast. The last set
(calibrated through the final data year) is the operational parameter set
for the coming year.

Outputs (under output/annual/<model>/):
  <site>_hindcast.csv       stitched daily out-of-sample predictions + states
  <site>_params.csv         parameters and calibration KGE for every target year
  operational_params.csv    parameters for the next (unobserved) year, all basins
  hindcast_summary.csv      per-basin hindcast skill (KGE, r, alpha, beta, NSE, bias)
  hindcast_by_year.csv      per-basin, per-year KGE and volume error

Usage:  python 07_annual_update.py [SITE_ID ...]         (IWB_MODEL=gr4j to use GR4J)
"""
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import json
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import END, MIN_OBS_YEARS, MODEL, OUT_DIR, PROC_DIR, WARMUP, ensure_dirs, selected_sites
from models import PARAMS, model_inputs, simulate

cal = importlib.import_module("04_calibrate")

FIRST_TARGET = 2016
CAL_START = "2002-01-01"
LAST_DATA_YEAR = int(END[:4])
# Cheaper than the one-off calibration, since each basin is calibrated ~11 times.
DE = dict(maxiter=150, popsize=15)
ANN_DIR = OUT_DIR / "annual" / MODEL
# (objective recorded in hindcast_summary.csv)


def run_site(site):
    df = pd.read_csv(PROC_DIR / f"{site}.csv", parse_dates=["date"], index_col="date")
    attrs = json.loads((PROC_DIR / f"{site}_attrs.json").read_text())
    inputs = model_inputs(df, attrs)
    obs = df.q_obs_mm.values
    years = df.index.year.values

    rows, pieces = [], []
    for y in range(FIRST_TARGET, LAST_DATA_YEAR + 2):
        window = (df.index >= CAL_START) & (years < y)
        n_obs = int((window & ~np.isnan(obs)).sum())
        if n_obs < MIN_OBS_YEARS * 365:
            continue
        p = cal.calibrate(inputs, obs, window, **DE)
        st = simulate(inputs, p, return_states=True)
        row = {"target_year": y, "cal_start": CAL_START[:4], "cal_end": y - 1,
               "cal_obs_days": n_obs, "KGE_calib": cal.kge(obs[window], st["q"][window]),
               "obj_calib": cal.objective(obs[window & ~np.isnan(obs)], st["q"][window & ~np.isnan(obs)]),
               **p}
        if y <= LAST_DATA_YEAR:
            m = years == y
            o = obs[m]
            if np.isfinite(o).sum() >= 300:
                row["KGE_pred"] = cal.kge(o, st["q"][m])
                ok = ~np.isnan(o)
                row["vol_err_pct"] = 100 * (st["q"][m][ok].sum() - o[ok].sum()) / o[ok].sum()
            piece = cal.sim_frame(df, st, "prediction")[m]
            piece["param_year"] = y
            pieces.append(piece)
        rows.append(row)

    if not rows:
        return {"site": site, "name": attrs["name"], "status": "skipped: short record"}
    pd.DataFrame(rows).round(5).to_csv(ANN_DIR / f"{site}_params.csv", index=False)
    out = {"site": site, "name": attrs["name"], "status": "ok", "objective": cal.OBJECTIVE,
           "operational_for": rows[-1]["target_year"]}
    if pieces:
        hind = pd.concat(pieces)
        hind.round(4).to_csv(ANN_DIR / f"{site}_hindcast.csv")
        sc = cal.scores(hind.q_obs_mm.values, hind.q_sim_mm.values)
        out.update({f"{k}_hindcast": v for k, v in sc.items()})
        out["hindcast_years"] = f"{hind.index.year.min()}-{hind.index.year.max()}"
    print(f"[{site}] hindcast KGE {out.get('KGE_hindcast', np.nan):.2f}  "
          f"({out.get('hindcast_years', '-')})  {attrs['name'][:40]}", flush=True)
    return out


def main():
    ensure_dirs()
    ANN_DIR.mkdir(parents=True, exist_ok=True)
    summ = pd.read_csv(OUT_DIR / "calibration_summary.csv", dtype={"site": str}).set_index("site")
    sites = [s for s in selected_sites(sys.argv)
             if s in summ.index and summ.loc[s, "status"] == "ok"]
    print(f"Annual updating, CemaNeige-{MODEL.upper()}: {len(sites)} basins, "
          f"targets {FIRST_TARGET}-{LAST_DATA_YEAR + 1}, {cal.N_WORKERS} workers", flush=True)
    with Pool(cal.N_WORKERS) as pool:
        res = pool.map(run_site, sites, chunksize=1)
    summary = pd.DataFrame(res).set_index("site")
    summary.round(4).to_csv(ANN_DIR / "hindcast_summary.csv")

    op, by_year = [], []
    for s in summary[summary.status == "ok"].index:
        p = pd.read_csv(ANN_DIR / f"{s}_params.csv")
        op.append({"site": s, **p.iloc[-1][["target_year", "cal_end"] + PARAMS].to_dict()})
        if "KGE_pred" in p:
            by_year.append(p[["target_year", "KGE_pred", "vol_err_pct"]].assign(site=s))
    pd.DataFrame(op).to_csv(ANN_DIR / "operational_params.csv", index=False)
    if by_year:
        pd.concat(by_year).to_csv(ANN_DIR / "hindcast_by_year.csv", index=False)

    ok = summary[summary.status == "ok"]
    print(f"\nOut-of-sample hindcast ({len(ok)} basins): median KGE "
          f"{ok.KGE_hindcast.median():.3f}, NSE {ok.NSE_hindcast.median():.3f}, "
          f"bias {ok.PBIAS_hindcast.median():+.1f}%")
    print(f"Wrote {ANN_DIR}")


if __name__ == "__main__":
    main()
