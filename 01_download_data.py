"""Download raw inputs for every selected basin.

Sources
  - USGS Water Data OGC API : gauge metadata + daily mean discharge (cfs)
  - USGS NLDI               : upstream basin boundary + NHDPlus COMID of the gauge
  - gridMET (4 km, daily)   : pr, tmmn, tmmx, pet over the bounding box of all basins
  - NOAA AORC v1.1 (1 km)   : hourly precipitation -> daily (CST) totals, same bbox
  - US Census TIGERweb      : Iowa state boundary for maps

Usage:  python 01_download_data.py [SITE_ID ...]
Downloads are cached in data/raw/ and skipped if already present.
"""
import json
import sys
import time

import pandas as pd
import requests

from common import (AORC_BUCKET, END, GRIDMET_NCSS, GRIDMET_VARS, LOCAL_UTC_OFFSET_H,
                    MAX_AREA_KM2, RAW_DIR,
                    SITES_FILE, SQMI_TO_KM2, START, ensure_dirs, selected_sites)

USGS_API = "https://api.waterdata.usgs.gov/ogcapi/v1/collections"
NLDI_API = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
TIGER_API = ("https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/"
             "State_County/MapServer/0/query")
BBOX_PAD_DEG = 0.1

session = requests.Session()


def get(url, params=None, retries=4, timeout=300):
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == retries - 1 or getattr(e.response, "status_code", 0) == 404:
                raise
            wait = 5 * 2 ** attempt
            print(f"    retry in {wait}s ({e})")
            time.sleep(wait)


def download_site_metadata(site, out):
    if not out.exists():
        d = get(f"{USGS_API}/monitoring-locations/items/USGS-{site}", {"f": "json"}).json()
        out.write_text(json.dumps(d))
    return json.loads(out.read_text())["properties"]


def download_basin(site, site_dir):
    out = site_dir / "basin.geojson"
    if not out.exists():
        d = get(f"{NLDI_API}/USGS-{site}/basin", {"f": "json"}).json()
        out.write_text(json.dumps(d))
    feat = site_dir / "nldi_feature.json"
    if not feat.exists():
        d = get(f"{NLDI_API}/USGS-{site}", {"f": "json"}).json()
        feat.write_text(json.dumps(d))


def download_streamflow(site, out):
    if out.exists():
        return
    url = f"{USGS_API}/daily/items"
    params = {
        "monitoring_location_id": f"USGS-{site}",
        "parameter_code": "00060",      # discharge
        "statistic_id": "00003",        # daily mean
        "time": f"{START}/{END}",
        "limit": 10000,
        "f": "json",
    }
    rows = []
    while url:
        d = get(url, params).json()
        for f in d["features"]:
            p = f["properties"]
            rows.append((p["time"], p["value"], p["approval_status"],
                         ",".join(p.get("qualifier") or [])))
        url = next((l["href"] for l in d["links"] if l["rel"] == "next"), None)
        params = None  # the "next" link already carries the query
    df = pd.DataFrame(rows, columns=["date", "q_cfs", "approval", "qualifier"])
    df["q_cfs"] = pd.to_numeric(df["q_cfs"], errors="coerce")
    df.sort_values("date").to_csv(out, index=False)


def basins_bbox(sites):
    lons, lats = [], []
    for s in sites:
        f = RAW_DIR / s / "basin.geojson"
        if not f.exists():
            continue
        for feat in json.loads(f.read_text())["features"]:
            g = feat["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for p in polys:
                lons += [c[0] for c in p[0]]
                lats += [c[1] for c in p[0]]
    return (min(lons) - BBOX_PAD_DEG, min(lats) - BBOX_PAD_DEG,
            max(lons) + BBOX_PAD_DEG, max(lats) + BBOX_PAD_DEG)


def download_gridmet(bbox):
    """One NetCDF per variable per year over bbox (west, south, east, north).
    Re-downloads a variable-year if the cached file does not cover bbox."""
    gdir = RAW_DIR / "gridmet"
    gdir.mkdir(exist_ok=True)
    meta_f = gdir / "bbox.json"
    old = json.loads(meta_f.read_text()) if meta_f.exists() else None
    if old and not (old[0] <= bbox[0] and old[1] <= bbox[1]
                    and old[2] >= bbox[2] and old[3] >= bbox[3]):
        print("  basin extent grew: refreshing gridMET files")
        for f in gdir.glob("*.nc"):
            f.unlink()
        old = None
    bbox = old or bbox
    meta_f.write_text(json.dumps(bbox))

    years = range(int(START[:4]), int(END[:4]) + 1)
    for var, long in GRIDMET_VARS.items():
        for yr in years:
            out = gdir / f"{var}_{yr}.nc"
            if out.exists():
                continue
            print(f"  gridMET {var} {yr}", flush=True)
            r = get(GRIDMET_NCSS.format(var=var), {
                "var": long, "west": bbox[0], "south": bbox[1], "east": bbox[2],
                "north": bbox[3], "disableProjSubset": "on", "horizStride": 1,
                "time_start": f"{yr}-01-01T00:00:00Z",
                "time_end": f"{yr}-12-31T00:00:00Z", "timeStride": 1,
                "accept": "netcdf"})
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.rename(out)


def download_aorc(bbox):
    """Daily AORC precipitation over bbox, one NetCDF per source year.

    Each file holds the sums over *local* days of that year's hourly values,
    plus the number of hours contributing to each day. Days straddling two
    source years (Dec 31 / Jan 1 in local time) appear partially in both
    files and are completed by summing across files in 02_prep_data.py.
    """
    import numpy as np
    import s3fs
    import xarray as xr

    adir = RAW_DIR / "aorc"
    adir.mkdir(exist_ok=True)
    fs = s3fs.S3FileSystem(anon=True)
    # Hour ending at t covers [t-1h, t); shift to local standard time.
    shift = np.timedelta64(1 - LOCAL_UTC_OFFSET_H, "h")
    for yr in range(int(START[:4]), int(END[:4]) + 2):   # +1: Dec 31 local spills into next UTC year
        out = adir / f"aorc_daily_{yr}.nc"
        if out.exists() or not fs.exists(f"{AORC_BUCKET}/{yr}.zarr"):
            continue
        print(f"  AORC {yr}", flush=True)
        ds = xr.open_zarr(s3fs.S3Map(f"{AORC_BUCKET}/{yr}.zarr", s3=fs, check=False),
                          consolidated=True)
        v = ds["APCP_surface"].sel(latitude=slice(bbox[1], bbox[3]),
                                   longitude=slice(bbox[0], bbox[2]))
        if yr > int(END[:4]):
            v = v.sel(time=slice(None, f"{yr}-01-01T{-LOCAL_UTC_OFFSET_H:02d}:00"))
        day = (v.time - shift).dt.floor("D").rename("day")
        daily = v.groupby(day).sum().astype("float32").compute()
        nhours = v.time.groupby(day).count().rename("nhours")
        xr.Dataset({"precip": daily, "nhours": nhours}).to_netcdf(
            out.with_suffix(".part"), format="NETCDF4",
            encoding={"precip": {"zlib": True, "complevel": 4}})
        out.with_suffix(".part").rename(out)


def download_iowa_boundary():
    out = RAW_DIR / "iowa_boundary.geojson"
    if out.exists():
        return
    r = get(TIGER_API, {"where": "STUSAB='IA'", "outFields": "NAME",
                        "f": "geojson", "maxAllowableOffset": 0.005})
    out.write_text(r.text)


def mark_excluded(updates):
    """Record exclusions found here (area, missing basin) in data/sites.csv."""
    if not updates:
        return
    df = pd.read_csv(SITES_FILE, dtype={"site_no": str})
    df["excluded"] = df["excluded"].fillna("")
    for site, why in updates.items():
        df.loc[df.site_no == site, "excluded"] = why
    df.to_csv(SITES_FILE, index=False)


def main():
    ensure_dirs()
    print("Iowa state boundary")
    download_iowa_boundary()
    sites = selected_sites(sys.argv)
    excluded = {}
    for site in sites:
        site_dir = RAW_DIR / site
        site_dir.mkdir(exist_ok=True)
        print(f"[{site}]", flush=True)
        meta = download_site_metadata(site, site_dir / "site.json")
        area = (meta.get("drainage_area") or 0) * SQMI_TO_KM2
        if area > MAX_AREA_KM2:
            excluded[site] = f"drainage area {area:.0f} km2 > {MAX_AREA_KM2}"
            print("  skipped:", excluded[site])
            continue
        if area == 0:
            excluded[site] = "no USGS drainage area"
            print("  skipped:", excluded[site])
            continue
        try:
            download_basin(site, site_dir)
        except requests.HTTPError as e:
            excluded[site] = "no NLDI basin"
            print(f"  skipped: {excluded[site]} ({e})")
            continue
        download_streamflow(site, site_dir / "streamflow.csv")
    mark_excluded(excluded)

    kept = [s for s in sites if s not in excluded]
    bbox = basins_bbox(kept)
    print(f"gridMET bbox (W,S,E,N) = {tuple(round(v, 2) for v in bbox)}")
    download_gridmet(bbox)
    download_aorc(bbox)
    print("Done.")


if __name__ == "__main__":
    main()
