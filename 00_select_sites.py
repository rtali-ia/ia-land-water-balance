"""Select model catchments: IWQIS nitrate sites co-located with a USGS
discharge gauge, excluding big-river mainstems and reservoir-regulated reaches.

Inputs (read-only, from the iowa-nitrates project):
  data/raw/iwqis-v2/measures.csv      which IWQIS sites carry nitrate / discharge
  data/raw/iwqis-v2/sites.csv         IWQIS site -> co-located USGS gauge
  data/raw/iwqis/bulk_sites.parquet   IWQIS site coordinates
  data/raw/usgs_nwis/discharge_sites.parquet   Iowa USGS discharge gauges

Output: data/sites.csv

Note: IWQIS does not measure discharge itself; its discharge series are the
USGS 00060 record (method "usgs.00060"). Discharge is therefore taken from
USGS, and IWQIS defines *which* outlets are modelled (the ones with
high-frequency nitrate sensors).
"""
import re

import numpy as np
import pandas as pd

from common import NITRATES_DIR, SITES_FILE

RAW = NITRATES_DIR / "data" / "raw"
MATCH_RADIUS_KM = 1.5

# Mainstems of the Mississippi/Missouri, and gauges below Saylorville,
# Red Rock or Coralville reservoirs, where a lumped rainfall-runoff model
# cannot reproduce managed releases.
EXCLUDE = {
    "06486000": "Missouri mainstem", "06610000": "Missouri mainstem",
    "06813500": "Missouri mainstem", "06934500": "Missouri mainstem",
    "05420460": "Mississippi mainstem", "05420500": "Mississippi mainstem",
    "07020500": "Mississippi mainstem", "07020850": "Mississippi mainstem",
    "05482000": "below Saylorville Reservoir",
    "05490500": "below Red Rock Reservoir",
    "05453520": "below Coralville Reservoir",
    "05454500": "below Coralville Reservoir",
    "05465500": "below Coralville Reservoir",
    "05455700": "below Coralville Reservoir",
    "05481650": "below Saylorville Reservoir",
    "05489500": "below Red Rock Reservoir",
    "05411950": "karst spring, no surface catchment",
}


def iwqis_colloc():
    """IWQIS site uid -> first co-located gauge id ('' if none).
    sites.csv holds unquoted Postgres arrays like {a,b}, so parse by regex."""
    out = {}
    lines = (RAW / "iwqis-v2" / "sites.csv").read_text().splitlines()[1:]
    for line in lines:
        uid = line.split(",")[0].strip()
        arrays = re.findall(r"\{([^}]*)\}", line)
        out[uid] = arrays[0].split(",")[0] if arrays else ""
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def main():
    meas = pd.read_csv(RAW / "iwqis-v2" / "measures.csv", dtype=str)
    nitrate_sites = sorted(set(meas.loc[meas.param_uid == "nitrate_con", "site_uid"].str.strip()))
    iw = pd.read_parquet(RAW / "iwqis" / "bulk_sites.parquet").set_index("station_id")
    gauges = pd.read_parquet(RAW / "usgs_nwis" / "discharge_sites.parquet")
    gauges = gauges[["site_no", "station_nm", "dec_lat_va", "dec_long_va"]].set_index("site_no")
    colloc = iwqis_colloc()

    rows = []
    for uid in nitrate_sites:
        if uid not in iw.index:
            continue
        s = iw.loc[uid]
        gauge, how = None, None
        if uid in gauges.index:
            gauge, how = uid, "iwqis id is usgs id"
        elif colloc.get(uid, "") in gauges.index:
            gauge, how = colloc[uid], "iwqis colloc field"
        else:
            d = haversine_km(s.lat, s.lon, gauges.dec_lat_va.values, gauges.dec_long_va.values)
            i = int(np.nanargmin(d))
            if d[i] <= MATCH_RADIUS_KM:
                gauge, how = gauges.index[i], f"nearest gauge {d[i]:.2f} km"
        if gauge is None:
            continue
        rows.append({"site_no": gauge, "iwqis_id": uid, "iwqis_name": s["name"],
                     "usgs_name": gauges.loc[gauge, "station_nm"],
                     "lat": gauges.loc[gauge, "dec_lat_va"],
                     "lon": gauges.loc[gauge, "dec_long_va"], "match": how})

    df = pd.DataFrame(rows).drop_duplicates("site_no")
    # IWQIS drainage areas are unreliable (mixed units, placeholders), so the
    # size screen (MAX_AREA_KM2) is applied in step 01 from USGS metadata.
    df["excluded"] = df.site_no.map(EXCLUDE).fillna("")
    df = df.sort_values("site_no")
    SITES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SITES_FILE, index=False)

    keep = df[df.excluded == ""]
    print(f"{len(nitrate_sites)} IWQIS nitrate sites -> {len(df)} matched to a USGS "
          f"discharge gauge -> {len(keep)} kept\n")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(df[["site_no", "iwqis_id", "usgs_name", "match", "excluded"]].to_string(index=False))


if __name__ == "__main__":
    main()
