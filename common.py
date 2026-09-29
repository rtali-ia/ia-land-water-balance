"""Shared configuration and helpers for the Iowa land-water balance model."""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "output"
SITES_FILE = ROOT / "data" / "sites.csv"

# Sister project holding IWQIS, StreamCat, tile drainage, CDL, NHDPlus (read-only).
NITRATES_DIR = Path(os.environ.get("IOWA_NITRATES_DIR",
                                   Path.home() / "Documents" / "iowa-nitrates"))

# Study period (gridMET covers 1979-present).
START = "2000-01-01"
END = "2025-12-31"
WARMUP = ("2000-01-01", "2001-12-31")   # spin-up, excluded from scoring
CALIB = ("2002-01-01", "2015-12-31")
VALID = ("2016-01-01", "2025-12-31")

# Sites with shorter records are split chronologically in half instead
# (see 04_calibrate.py); below this many observed years a site is skipped.
MIN_OBS_YEARS = 4

# Largest basin kept (km2). Bigger rivers are dominated by routing/regulation.
MAX_AREA_KM2 = 25000

# gridMET: THREDDS NetCDF-subset service, one request per variable per year.
GRIDMET_NCSS = ("http://thredds.northwestknowledge.net:8080/thredds/ncss/"
                "agg_met_{var}_1979_CurrentYear_CONUS.nc")
GRIDMET_VARS = {                # short name -> variable name inside the file
    "pr": "precipitation_amount",           # mm/day
    "tmmn": "daily_minimum_temperature",    # K
    "tmmx": "daily_maximum_temperature",    # K
    "pet": "daily_mean_reference_evapotranspiration_grass",  # mm/day, ASCE grass ETo
}

# Rainfall-runoff model: "gr6j" (default; exponential store carries multi-year
# drought memory) or "gr4j". Both use the CemaNeige snow routine.
MODEL = os.environ.get("IWB_MODEL", "gr6j").lower()

# Calibration objective: KGE on sqrt(Q) weighs floods and low flows evenly
# (tile-drainage baseflow carries most of Iowa's nitrate); "kge" = raw Q.
OBJECTIVE = os.environ.get("IWB_OBJECTIVE", "kge_sqrt").lower()

# Crop coefficient applied to gridMET grass reference ET. Tested in
# experiments/improvements.py: a per-basin calibrated value has median 0.75
# (IQR 0.73-0.80); 0.78 (the iowa-nitrates value) improved low-flow skill
# (log-NSE 0.55 -> 0.64) and cut the model's water import from 87 to 16 mm/yr.
PET_CROP_COEF = float(os.environ.get("IWB_KC", "0.78"))

# Precipitation source: "aorc" (NOAA AORC v1.1, 1 km hourly -> daily) or
# "gridmet". AORC is what NWM (and the iowa-nitrates project) uses, and
# gridMET precipitation shows a +4-6% step after 2016 relative to Daymet.
PRECIP_SOURCE = "aorc"
AORC_BUCKET = "noaa-nws-aorc-v1-1-1km"
# AORC is hourly in UTC, each value the accumulation of the hour ending at its
# timestamp. Daily totals are summed over local standard-time days (CST,
# UTC-6) to match USGS daily mean discharge.
LOCAL_UTC_OFFSET_H = -6

CFS_TO_M3S = 0.0283168466
SQMI_TO_KM2 = 2.58998811


def ensure_dirs():
    for d in (RAW_DIR, PROC_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_sites(include_excluded=False):
    """{site_no: name} from data/sites.csv (built by 00_select_sites.py)."""
    if not SITES_FILE.exists():
        raise SystemExit("data/sites.csv missing - run 00_select_sites.py first")
    with SITES_FILE.open() as f:
        rows = list(csv.DictReader(f))
    return {r["site_no"]: r["usgs_name"] for r in rows
            if include_excluded or not r["excluded"]}


def selected_sites(argv):
    """Sites given on the command line, or all kept sites."""
    sites = load_sites()
    ids = argv[1:] or list(sites)
    unknown = [s for s in ids if s not in sites]
    if unknown:
        raise SystemExit(f"Unknown site(s) {unknown}; see data/sites.csv")
    return ids
