"""Physical attributes for each gauged basin (used to regionalize parameters).

  tile_frac        USGS 2012 subsurface-drainage raster (30 m) over the NLDI polygon
  soils, NLCD      EPA StreamCat *watershed* metrics at the gauge COMID
                   (clay, sand, permeability, water-table depth, OM, BFI, ...)
  corn/soy, slope  CDL 2013-2022 and slope from the iowa-nitrates catchment
                   tables, area-weighted over the Iowa part of the basin
  climate          gridMET 2002-2025 means from data/processed/<site>.csv

Output: data/processed/basin_attributes.csv

Usage:  python 03_basin_attributes.py
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask

from common import NITRATES_DIR, PROC_DIR, RAW_DIR, load_sites

ND = NITRATES_DIR / "data"
TILE_TIF = next((ND / "raw" / "tile_drainage").rglob("SubsurfaceDrainExtentMW_2012.tif"))

# StreamCat metrics used (suffix "ws" = whole upstream watershed,
# "cat" = local NHDPlus catchment only; the regionalization uses the same
# names without suffix).
STREAMCAT = {
    "clay": "clay", "sand": "sand", "perm": "perm", "wtdep": "wtdep",
    "om": "om", "bfi": "bfi", "hydrlcond": "hydrlcond", "kffact": "kffact",
    "pctcrop2019": "pct_crop", "pcthay2019": "pct_hay", "pctdecid2019": "pct_forest",
    "precip8110": "prism_precip", "tmean8110": "prism_tmean",
}


def tile_fraction(geom_wgs84):
    """Share of basin area mapped as subsurface (tile) drained."""
    with rasterio.open(TILE_TIF) as src:
        g = gpd.GeoSeries([geom_wgs84], crs=4326).to_crs(src.crs)
        arr, _ = mask(src, list(g), crop=True, filled=True, nodata=0)
        a = arr[0].astype("float64")
        a[(a < 0) | ~np.isfinite(a)] = 0.0        # nodata = not drained
        return float(a.sum() / g.area.iloc[0])      # pixel value = drained m2


def main():
    sites = load_sites()
    cov = pd.read_parquet(ND / "processed" / "covariates_static.parquet").set_index("comid")
    xwalk = pd.read_parquet(ND / "interim" / "gauge_reach_crosswalk.parquet")
    xwalk = xwalk.set_index("site_no")["comid"]
    cats = gpd.read_parquet(ND / "raw" / "nhdplus" / "catchments.parquet")
    cats = cats.rename(columns={"featureid": "comid"})[["comid", "areasqkm", "geometry"]]
    cats["pt"] = cats.geometry.representative_point()
    crops = pd.read_parquet(ND / "interim" / "crop_fractions_by_catchment.parquet")
    crops = crops.groupby("comid")[["pct_corn", "pct_soy"]].mean()
    slope = pd.read_parquet(ND / "interim" / "slope_by_catchment.parquet").set_index("comid")

    rows = []
    for site, name in sites.items():
        attrs_f = PROC_DIR / f"{site}_attrs.json"
        if not attrs_f.exists():
            continue
        at = json.loads(attrs_f.read_text())
        geom = gpd.read_file(RAW_DIR / site / "basin.geojson").to_crs(4326).union_all()
        r = {"site": site, "name": name, "area_km2": at["drainage_area_km2"],
             "log_area": np.log10(at["drainage_area_km2"])}

        r["tile_frac"] = tile_fraction(geom)

        # StreamCat: prefer the NLDI COMID, fall back to the nitrates crosswalk.
        comid = at.get("comid")
        if comid not in cov.index and site in xwalk.index:
            comid = int(xwalk[site])
        r["comid"] = comid
        if comid in cov.index:
            for k, v in STREAMCAT.items():
                r[v] = cov.at[comid, f"{k}ws"]
        # Iowa-part catchment aggregates (CDL, slope).
        inside = cats[cats.pt.within(geom)]
        w = inside.set_index("comid")["areasqkm"]
        r["iowa_coverage"] = w.sum() / at["drainage_area_km2"]
        if len(w):
            c = crops.reindex(w.index)
            ok = c.notna().all(axis=1)
            r["pct_corn"] = np.average(c.pct_corn[ok], weights=w[ok]) if ok.any() else np.nan
            r["pct_soy"] = np.average(c.pct_soy[ok], weights=w[ok]) if ok.any() else np.nan
            s = slope.reindex(w.index)["slope_pct"]
            r["slope_pct"] = np.average(s[s.notna()], weights=w[s.notna()])

        f = pd.read_csv(PROC_DIR / f"{site}.csv", parse_dates=["date"], index_col="date")
        f = f.loc["2002":"2025"]
        yrs = len(f) / 365.25
        r["p_mm"] = f.precip_mm.sum() / yrs
        r["pet_mm"] = f.pet_mm.sum() / yrs
        r["aridity"] = r["pet_mm"] / r["p_mm"]
        r["snow_frac"] = f.precip_mm[f.temp_c < 0].sum() / f.precip_mm.sum()
        r["t_mean"] = f.temp_c.mean()
        # Observed signature (diagnostic only, never a predictor).
        q = f.q_obs_mm.dropna()
        r["obs_runoff_ratio"] = q.mean() / f.precip_mm[q.index].mean() if len(q) else np.nan
        rows.append(r)
        print(f"[{site}] tile={r['tile_frac']:.2f} crop={r.get('pct_crop', np.nan):5.1f}% "
              f"clay={r.get('clay', np.nan):4.1f}% IA-cover={r['iowa_coverage']:.2f} "
              f"P={r['p_mm']:.0f} RR={r['obs_runoff_ratio']:.2f}  {name[:38]}")

    df = pd.DataFrame(rows).set_index("site")
    df.round(4).to_csv(PROC_DIR / "basin_attributes.csv")
    print(f"\nWrote {PROC_DIR / 'basin_attributes.csv'} ({len(df)} basins)")


if __name__ == "__main__":
    main()
