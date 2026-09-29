"""Turn raw downloads into model-ready daily forcing for each basin.

Forcing is the basin average of each gridded product, weighting every cell by
the share of the NLDI basin polygon it covers:
  precipitation  AORC v1.1 (1 km, daily CST totals)   [PRECIP_SOURCE = "aorc"]
  temperature    gridMET tmmn / tmmx (4 km)
  PET            gridMET grass reference ET (4 km); Oudin kept for comparison

Output: data/processed/<site>.csv with columns
    precip_mm, precip_gridmet_mm, tmin_c, tmax_c, temp_c, pet_mm, pet_oudin_mm, q_obs_mm
and data/processed/<site>_attrs.json (areas, elevation, record length).

Usage:  python 02_prep_data.py [SITE_ID ...]
"""
import json
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy import sparse

from common import (CFS_TO_M3S, END, GRIDMET_VARS, PRECIP_SOURCE, PROC_DIR,
                    RAW_DIR, SQMI_TO_KM2, START, ensure_dirs, selected_sites)

FT_TO_M = 0.3048


def basin_geometry(site):
    return gpd.read_file(RAW_DIR / site / "basin.geojson").to_crs(4326).union_all()


def cell_weights(basins, lat, lon):
    """Sparse W [n_basins, n_cells]: share of each basin's area in each cell."""
    dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
    lon2d, lat2d = np.meshgrid(lon, lat)            # cell centres, row-major
    boxes = shapely.box(lon2d.ravel() - dlon / 2, lat2d.ravel() - dlat / 2,
                        lon2d.ravel() + dlon / 2, lat2d.ravel() + dlat / 2)
    rows, cols, vals = [], [], []
    tree = shapely.STRtree(boxes)
    for i, geom in enumerate(basins):
        idx = tree.query(geom, predicate="intersects")
        # Scale by cos(lat) so cells are weighted by true area.
        a = shapely.area(shapely.intersection(boxes[idx], geom)) * np.cos(np.radians(lat2d.ravel()[idx]))
        rows += [i] * len(idx)
        cols += list(idx)
        vals += list(a / a.sum())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(len(basins), boxes.size))


def weighted_mean(arr, W):
    """Basin means of arr [n_days, n_cells], renormalising over non-NaN cells."""
    num = (W @ np.nan_to_num(arr).T).T
    den = (W @ (~np.isnan(arr)).astype(float).T).T
    return num / den


def gridmet_basin_means(sites):
    """{short var: DataFrame[date x site]} of basin-average gridMET."""
    gdir = RAW_DIR / "gridmet"
    basins = [basin_geometry(s) for s in sites]
    years = range(int(START[:4]), int(END[:4]) + 1)
    out, W = {}, None
    for var, long in GRIDMET_VARS.items():
        parts = []
        for yr in years:
            with xr.open_dataset(gdir / f"{var}_{yr}.nc") as ds:
                da = ds[long]
                if W is None:
                    W = cell_weights(basins, da.lat.values, da.lon.values)
                arr = da.transpose("day", "lat", "lon").values.reshape(da.sizes["day"], -1)
                parts.append(pd.DataFrame(weighted_mean(arr, W),
                                          index=pd.DatetimeIndex(da.day.values), columns=sites))
        out[var] = pd.concat(parts)
    return out


def aorc_basin_means(sites):
    """DataFrame[date x site] of basin-average daily AORC precipitation (mm).

    Local days straddling two source years are summed across files; a day
    with fewer than 24 source hours (only the final day, whose last UTC hours
    fall in a year not yet published) is scaled up to a 24-hour total.
    """
    files = sorted((RAW_DIR / "aorc").glob("aorc_daily_*.nc"))
    basins = [basin_geometry(s) for s in sites]
    W, parts, hours = None, [], []
    for f in files:
        with xr.open_dataset(f) as ds:
            if W is None:
                W = cell_weights(basins, ds.latitude.values, ds.longitude.values)
            da = ds["precip"].transpose("day", "latitude", "longitude")
            arr = da.values.reshape(da.sizes["day"], -1).astype("float64")
            idx = pd.DatetimeIndex(ds.day.values)
            parts.append(pd.DataFrame(weighted_mean(arr, W), index=idx, columns=sites))
            hours.append(pd.Series(ds.nhours.values, index=idx))
    p = pd.concat(parts).groupby(level=0).sum()
    h = pd.concat(hours).groupby(level=0).sum()
    p = p.loc[START:END]
    h = h.loc[START:END]
    short = h[h < 24]
    if len(short):
        print(f"  AORC: {len(short)} day(s) with <24 h, scaled: "
              + ", ".join(f"{d.date()} ({n} h)" for d, n in short.items()))
        p.loc[short.index] = p.loc[short.index].mul(24 / short, axis=0)
    return p


def extraterrestrial_radiation(doy, lat_deg):
    """FAO-56 Eq. 21, MJ m-2 day-1."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(delta), -1, 1))
    return (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(phi) * np.sin(delta) + np.cos(phi) * np.cos(delta) * np.sin(ws))


def pet_oudin(temp_c, doy, lat_deg):
    """Oudin et al. (2005) PET [mm/day]; kept for sensitivity runs."""
    re = extraterrestrial_radiation(doy, lat_deg)
    return np.where(temp_c + 5 > 0, re / 2.45 * (temp_c + 5) / 100, 0.0)


def main():
    ensure_dirs()
    sites = [s for s in selected_sites(sys.argv) if (RAW_DIR / s / "basin.geojson").exists()]
    print(f"Averaging gridMET over {len(sites)} basins ...", flush=True)
    met = gridmet_basin_means(sites)
    if PRECIP_SOURCE == "aorc":
        print("Averaging AORC precipitation ...", flush=True)
        aorc = aorc_basin_means(sites)
    dates = pd.date_range(START, END, freq="D")

    for site in sites:
        raw = RAW_DIR / site
        meta = json.loads((raw / "site.json").read_text())
        props = meta["properties"]
        geom = basin_geometry(site)
        centroid = geom.centroid

        out = pd.DataFrame(index=dates)
        out.index.name = "date"
        out["precip_gridmet_mm"] = met["pr"][site].reindex(dates)
        out["precip_mm"] = (aorc[site].reindex(dates) if PRECIP_SOURCE == "aorc"
                            else out["precip_gridmet_mm"])
        out["tmin_c"] = met["tmmn"][site].reindex(dates) - 273.15
        out["tmax_c"] = met["tmmx"][site].reindex(dates) - 273.15
        out["temp_c"] = (out.tmin_c + out.tmax_c) / 2
        out["pet_mm"] = met["pet"][site].reindex(dates)
        out["pet_oudin_mm"] = pet_oudin(out.temp_c.values, out.index.dayofyear.values,
                                        centroid.y)
        gaps = out.isna().sum().sum()
        out = out.interpolate(limit=3)          # isolated missing days only
        assert not out.isna().any().any(), f"{site}: gaps in forcing"

        area_km2 = props["drainage_area"] * SQMI_TO_KM2
        q = pd.read_csv(raw / "streamflow.csv", parse_dates=["date"], index_col="date")
        q = q["q_cfs"].clip(lower=0).reindex(dates)
        out["q_obs_mm"] = q * CFS_TO_M3S * 86400 / (area_km2 * 1e6) * 1000

        nldi = json.loads((raw / "nldi_feature.json").read_text())["features"][0]["properties"]
        poly_km2 = gpd.GeoSeries([geom], crs=4326).to_crs(5070).area.iloc[0] / 1e6
        gauge_elev = props.get("altitude")
        attrs = {
            "site": site,
            "name": props["monitoring_location_name"],
            "comid": int(nldi["comid"]) if nldi.get("comid") else None,
            "gauge_lon": meta["geometry"]["coordinates"][0],
            "gauge_lat": meta["geometry"]["coordinates"][1],
            "centroid_lon": round(centroid.x, 4),
            "centroid_lat": round(centroid.y, 4),
            "drainage_area_km2": round(area_km2, 1),
            "nldi_polygon_area_km2": round(poly_km2, 1),
            # Single elevation layer: Iowa relief is ~100-150 m per basin, too
            # little for CemaNeige bands to matter. Gauge datum is used as the
            # reference height (only selects the solid-fraction method).
            "ref_elev_m": round(gauge_elev * FT_TO_M, 1) if gauge_elev else 300.0,
            "forcing_gap_days": int(gaps),
            "q_obs_days": int(out.q_obs_mm.notna().sum()),
            "q_first": str(out.q_obs_mm.first_valid_index().date()) if out.q_obs_mm.notna().any() else None,
            "q_last": str(out.q_obs_mm.last_valid_index().date()) if out.q_obs_mm.notna().any() else None,
        }
        out.round(4).to_csv(PROC_DIR / f"{site}.csv")
        (PROC_DIR / f"{site}_attrs.json").write_text(json.dumps(attrs, indent=2))

        yrs = out.loc["2002":"2025"]
        ann = yrs.resample("YE").sum(min_count=300).mean()
        print(f"[{site}] {attrs['name'][:40]:40s} A={area_km2:7.0f} km2 "
              f"(poly {poly_km2 / area_km2:5.2f}x)  P={ann.precip_mm:4.0f} "
              f"PET={ann.pet_mm:5.0f} Q={ann.q_obs_mm:4.0f} mm/yr  "
              f"Q days={attrs['q_obs_days']:5d} ({attrs['q_first']}..{attrs['q_last']})")


if __name__ == "__main__":
    main()
