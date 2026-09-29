"""Regionalize CemaNeige-GR4J parameters from basin attributes and transfer
them to every NHDPlus catchment in Iowa.

Methods, each scored by leave-one-out (LOO) on the held-out gauge's
validation period. Donor/training basins that are nested with the target
(share >10% of area) are excluded from its fold, so skill is not inflated by
upstream/downstream neighbours.

  calibrated    the site's own calibrated parameters (upper benchmark)
  regional      median parameters of the other basins (lower benchmark)
  regression    ridge regression of transformed parameters on attributes
  similarity    output-average of the k most attribute-similar donors
  proximity     output-average of the k nearest donors (centroid distance)

Outputs
  output/regionalization_loo.csv              per-site LOO KGE by method
  output/attribute_param_correlations.csv     Spearman rho, attributes vs parameters
  output/nhdplus_params.parquet               parameters for every Iowa NHDPlus catchment
                                              (best method), for the iowa-nitrates model

Usage:  python 05_regionalize.py
"""
import importlib
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (GRIDMET_VARS, MODEL, NITRATES_DIR, OUT_DIR, PRECIP_SOURCE,
                    PROC_DIR, RAW_DIR)
from models import LOG_PARAMS

cal = importlib.import_module("04_calibrate")

# Attributes available both for gauged basins (watershed) and for every
# NHDPlus catchment (local), so fitted relationships can be transferred.
PREDICTORS = ["tile_frac", "clay", "sand", "perm", "wtdep", "om", "bfi",
              "pct_crop", "pct_forest", "prism_precip", "prism_tmean",
              "aridity", "log_area"]
K_DONORS = 5
NEST_OVERLAP = 0.10
UB = {"CTG": 1.0, "Kf": 10.0}          # logit-transformed within [0, UB]


# ------------------------------------------------------ parameter transforms --
def to_space(p):
    """Parameter dict -> unbounded vector used for regression."""
    out = []
    for k in cal.PARAMS:
        v = p[k]
        if k in UB:
            f = np.clip(v / UB[k], 0.01, 0.99)
            out.append(np.log(f / (1 - f)))
        elif k in LOG_PARAMS:
            out.append(np.log(v))
        else:                       # x2, x5 can be negative
            out.append(v)
    return np.array(out)


def from_space(z):
    p = {}
    for k, v in zip(cal.PARAMS, z):
        if k in UB:
            p[k] = UB[k] / (1 + np.exp(-v))
        elif k in LOG_PARAMS:
            p[k] = float(np.exp(v))
        else:
            p[k] = float(v)
        lo, hi = cal.BOUNDS[k]
        p[k] = float(np.clip(p[k], lo, hi))
    return p


# ------------------------------------------------------------------ helpers --
def nested_pairs(sites):
    geoms = {s: gpd.read_file(RAW_DIR / s / "basin.geojson").to_crs(5070).union_all()
             for s in sites}
    nested = {s: set() for s in sites}
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            ga, gb = geoms[a], geoms[b]
            if not ga.intersects(gb):
                continue
            share = ga.intersection(gb).area / min(ga.area, gb.area)
            if share > NEST_OVERLAP:
                nested[a].add(b)
                nested[b].add(a)
    return nested


def fit_regression(X, Z):
    models = []
    for j in range(Z.shape[1]):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
        m.fit(X, Z[:, j])
        models.append(m)
    return models


def predict_regression(models, X):
    return np.column_stack([m.predict(X) for m in models])


def main():
    summ = pd.read_csv(OUT_DIR / "calibration_summary.csv", dtype={"site": str}).set_index("site")
    summ = summ[summ.status == "ok"]
    att = pd.read_csv(PROC_DIR / "basin_attributes.csv", dtype={"site": str}).set_index("site")
    sites = [s for s in summ.index if s in att.index and att.loc[s, PREDICTORS].notna().all()]
    att = att.loc[sites]
    params = {s: json.loads((OUT_DIR / "params" / f"{s}.json").read_text())["params"] for s in sites}
    Z = np.array([to_space(params[s]) for s in sites])
    X = att[PREDICTORS].values
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    cent = np.array([[json.loads((PROC_DIR / f"{s}_attrs.json").read_text())[k]
                      for k in ("centroid_lat", "centroid_lon")] for s in sites])
    nested = nested_pairs(sites)
    print(f"{len(sites)} basins in regionalization; "
          f"{sum(len(v) > 0 for v in nested.values())} have nested neighbours")

    # --- diagnostics: which attributes explain which parameters -------------
    corr = pd.DataFrame({k: [spearmanr(att[a], [params[s][k] for s in sites])[0]
                             for a in PREDICTORS] for k in cal.PARAMS}, index=PREDICTORS)
    corr.round(3).to_csv(OUT_DIR / "attribute_param_correlations.csv")
    print("\nSpearman rho, attribute vs calibrated parameter:")
    print(corr.round(2).to_string())

    # --- leave-one-out evaluation --------------------------------------------
    rows = []
    for i, s in enumerate(sites):
        train = [j for j, t in enumerate(sites) if t != s and t not in nested[s]]
        df = pd.read_csv(PROC_DIR / f"{s}.csv", parse_dates=["date"], index_col="date")
        a = json.loads((PROC_DIR / f"{s}_attrs.json").read_text())
        inputs = cal.model_inputs(df, a)
        _, val, _ = cal.split_periods(df)
        obs = df.q_obs_mm.values
        score = lambda q: cal.kge(obs[val], q[val])

        r = {"site": s, "name": summ.loc[s, "name"], "calibrated": summ.loc[s, "KGE_val"]}
        r["regional"] = score(cal.simulate(inputs, from_space(np.median(Z[train], axis=0))))

        models = fit_regression(X[train], Z[train])
        r["regression"] = score(cal.simulate(inputs, from_space(predict_regression(models, X[[i]])[0])))

        d_attr = np.linalg.norm(Xs[train] - Xs[i], axis=1)
        donors = [train[j] for j in np.argsort(d_attr)[:K_DONORS]]
        r["similarity"] = score(np.mean([cal.simulate(inputs, params[sites[j]]) for j in donors], axis=0))

        d_geo = np.hypot(cent[train, 0] - cent[i, 0],
                         (cent[train, 1] - cent[i, 1]) * np.cos(np.radians(cent[i, 0])))
        donors = [train[j] for j in np.argsort(d_geo)[:K_DONORS]]
        r["proximity"] = score(np.mean([cal.simulate(inputs, params[sites[j]]) for j in donors], axis=0))
        rows.append(r)
        print(f"[{s}] cal {r['calibrated']:.2f} | reg-mean {r['regional']:.2f} "
              f"ridge {r['regression']:.2f} simil {r['similarity']:.2f} "
              f"prox {r['proximity']:.2f}  {r['name'][:36]}", flush=True)

    loo = pd.DataFrame(rows).set_index("site")
    loo.round(4).to_csv(OUT_DIR / "regionalization_loo.csv")
    methods = ["regional", "regression", "similarity", "proximity"]
    print("\nLOO validation KGE (median / 25th pct):")
    for m in ["calibrated"] + methods:
        print(f"  {m:11s} {loo[m].median():.3f} / {loo[m].quantile(0.25):.3f}")
    # Transfer with donor similarity: it only ever assigns parameter sets that
    # were calibrated at a real gauge. Ridge regression extrapolates linearly
    # in attribute space (NHDPlus catchments are 1-5 km2 vs 12-20,000 km2
    # gauged basins) and produced out-of-range parameters (e.g. negative x2,
    # x1 pinned at its bound), so it is used only if clearly better in LOO.
    best = "similarity"
    if loo["regression"].median() > loo["similarity"].median() + 0.05:
        best = "regression"
    print(f"Transfer method for NHDPlus catchments: {best} "
          f"(LOO median: similarity {loo['similarity'].median():.3f}, "
          f"regression {loo['regression'].median():.3f})")

    # --- transfer to every NHDPlus catchment -----------------------------
    # Transfer the operational parameters (calibrated on all history, from
    # 07_annual_update.py) when available; the LOO test above necessarily
    # uses the fixed-split parameters so the held-out years stay unseen.
    op_f = OUT_DIR / "annual" / MODEL / "operational_params.csv"
    if op_f.exists():
        op = pd.read_csv(op_f, dtype={"site": str}).set_index("site")
        params = {s: (op.loc[s, cal.PARAMS].astype(float).to_dict() if s in op.index else params[s])
                  for s in sites}
        Z = np.array([to_space(params[s]) for s in sites])
        print(f"Transferring operational parameters from {op_f.relative_to(OUT_DIR.parent)}")
    transfer_to_nhdplus(best, X, Z, scaler, sites, params)


def mean_annual(files, var, first_year=2002):
    acc, n = None, 0
    for f in files:
        yr = int(f.stem.split("_")[-1])
        if yr < first_year or yr > 2025:
            continue
        with xr.open_dataset(f) as ds:
            s = ds[var].sum(ds[var].dims[0], skipna=False).load()
        acc = s if acc is None else acc + s
        n += 1
    return acc / n


def climatology():
    """Mean annual P and PET (2002-2025): P from the configured source, PET from gridMET."""
    pet = mean_annual(sorted((RAW_DIR / "gridmet").glob("pet_*.nc")), GRIDMET_VARS["pet"])
    if PRECIP_SOURCE == "aorc":
        p = mean_annual(sorted((RAW_DIR / "aorc").glob("aorc_daily_*.nc")), "precip")
        p = p.rename(latitude="lat", longitude="lon")
    else:
        p = mean_annual(sorted((RAW_DIR / "gridmet").glob("pr_*.nc")), GRIDMET_VARS["pr"])
    return p, pet


def transfer_to_nhdplus(method, X, Z, scaler, sites, params):
    ND = NITRATES_DIR / "data"
    cov = pd.read_parquet(ND / "processed" / "covariates_static.parquet").set_index("comid")
    cats = gpd.read_parquet(ND / "raw" / "nhdplus" / "catchments.parquet")
    cats = cats.rename(columns={"featureid": "comid"}).set_index("comid")
    cats = cats.loc[cats.index.intersection(cov.index)]
    pt = cats.geometry.representative_point()

    p, pet = climatology()
    sel = dict(lat=xr.DataArray(pt.y.values, dims="c"), lon=xr.DataArray(pt.x.values, dims="c"))
    ar = pet.sel(**sel, method="nearest").values / p.sel(**sel, method="nearest").values

    c = cov.loc[cats.index]
    Xc = pd.DataFrame({
        "tile_frac": c.tile_drain_frac, "clay": c.claycat, "sand": c.sandcat,
        "perm": c.permcat, "wtdep": c.wtdepcat, "om": c.omcat, "bfi": c.bficat,
        "pct_crop": c.pctcrop2019cat, "pct_forest": c.pctdecid2019cat,
        "prism_precip": c.precip8110cat, "prism_tmean": c.tmean8110cat,
        "aridity": ar, "log_area": np.log10(cats.areasqkm.clip(lower=0.01)),
    }, index=cats.index)[PREDICTORS]
    ok = Xc.notna().all(axis=1)
    Xc = Xc[ok]

    # Where a catchment lies outside the range of the gauged basins, the
    # prediction is an extrapolation; flag it.
    lo, hi = X.min(axis=0), X.max(axis=0)
    outside = ((Xc.values < lo) | (Xc.values > hi))
    out = pd.DataFrame(index=Xc.index)
    out["n_attrs_outside_gauged_range"] = outside.sum(axis=1)

    if method == "regression":
        models = fit_regression(X, Z)
        P = np.array([list(from_space(z).values()) for z in predict_regression(models, Xc.values)])
        for j, k in enumerate(cal.PARAMS):
            out[k] = P[:, j]
    else:
        Xs_g = scaler.transform(X)
        Xs_c = scaler.transform(Xc.values)
        d = ((Xs_c[:, None, :] - Xs_g[None, :, :]) ** 2).sum(axis=2)
        nn = np.argsort(d, axis=1)[:, :K_DONORS]
        for j in range(K_DONORS):
            out[f"donor{j + 1}"] = [sites[i] for i in nn[:, j]]
        # Median of donor parameters for convenience; for fidelity with the LOO
        # test, run all donors and average their flows.
        for k in cal.PARAMS:
            vals = np.array([[params[sites[i]][k] for i in row] for row in nn])
            out[k] = np.median(vals, axis=1)
    out["method"] = method
    out.to_parquet(OUT_DIR / "nhdplus_params.parquet")
    print(f"Wrote {OUT_DIR / 'nhdplus_params.parquet'}: {len(out)} catchments "
          f"({(~ok).sum()} lacked attributes); "
          f"{(out.n_attrs_outside_gauged_range > 0).mean():.0%} extrapolate on >=1 attribute")


if __name__ == "__main__":
    main()
