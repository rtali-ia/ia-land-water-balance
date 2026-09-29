# Iowa Water Balance Model

This is a model for simulating the water balance in Iowa based on precipitation, evapotranspiration, and runoff. The model uses historical climate data to estimate water availability and can be used for agricultural planning, water resource management, and environmental studies.

The model is **CemaNeige snow + GR6J**, with GR4J available for comparison, forced for 2000–2025 with daily **AORC v1.1** precipitation (the same product NWM and `iowa-nitrates` use) and **gridMET** temperature and grass reference ET. It is calibrated at every USGS gauge that shares a location with an **IWQIS** nitrate sensor. Its parameters are then transferred to all ~81,000 Iowa NHDPlus catchments using tile-drainage, soil, land-cover and climate attributes, for use by the sister project [`iowa-nitrates`](../iowa-nitrates).

## Objective

For each catchment, close the water balance **P = ET + Q + ΔS** at a daily time step. Streamflow Q at the outlet is the only measured term, so it is the calibration target. ET and ΔS are inferred once the model reproduces Q.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run_all.sh          # ~50 min first run: gridMET (~1.5 GB) + AORC (~1.4 GB) downloads, ~6 min calibration on 12 cores
```

Tile drainage, StreamCat, CDL, NHDPlus and IWQIS are read, never written, from `../iowa-nitrates/data`. Set `IOWA_NITRATES_DIR` to point elsewhere.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `00_select_sites.py` | Finds the IWQIS nitrate sites, matches each to a USGS discharge gauge (by the IWQIS id, its co-location field, or the nearest gauge within 1.5 km), and drops mainstems and reservoir-regulated gauges. Writes `data/sites.csv`. |
| 1 | `01_download_data.py` | Downloads USGS daily discharge and metadata (Water Data OGC API), the NLDI basin polygon and COMID, and gridMET `pr`/`tmmn`/`tmmx`/`pet` for 2000–2025. Also downloads AORC hourly precipitation from NOAA's public S3 bucket, summed to local-standard-time (CST) days to match USGS daily means. Both cover the bounding box of all basins. |
| 2 | `02_prep_data.py` | Area-averages AORC and gridMET over each basin, weighting cells by the share inside the polygon. Precipitation comes from `PRECIP_SOURCE` in `common.py`, and gridMET precipitation is kept as `precip_gridmet_mm`. Converts discharge from cfs to mm/day and computes Oudin PET as an alternative. |
| 3 | `03_basin_attributes.py` | Computes each basin's tile-drained fraction from the USGS 2012 30 m raster over the full polygon. Takes watershed soils (clay, sand, permeability, water-table depth, OM, BFI) and NLCD cover from StreamCat, CDL corn/soy and slope over the Iowa part, and gridMET climate. |
| 4 | `04_calibrate.py` | Calibrates each basin by differential evolution, maximising KGE on √Q, with a 2000–01 warm-up, 2002–15 calibration and 2016–25 validation. Gauges with short records are split in half chronologically. This fixed split is used to evaluate the model and the regionalization. |
| 7 | `07_annual_update.py` | **Operational scheme.** For each year Y, calibrates on all data from 2002 to Y−1 and predicts Y. Stitches the years into an out-of-sample 2016–2025 hindcast. The last set, calibrated on 2002–2025, is the operational parameter set for 2026 and is what gets transferred to NHDPlus catchments. |
| 5 | `05_regionalize.py` | Tests four regionalization methods by leave-one-out, excluding nested basins from each fold. Transfers attribute-similarity donors to every NHDPlus catchment and writes `output/nhdplus_params.parquet`. Regression is only used if it beats similarity by more than 0.05 KGE, because it extrapolates to impossible parameter values at NHDPlus scale. |
| 6 | `06_plot_results.py` | Draws the statewide maps and figures, the GR4J/GR6J annual-update comparison, and a dashboard for each basin. |

`models.py` holds the CemaNeige-GR4J kernel (rrmpg) and a numba CemaNeige-GR6J written from airGR's `frun_GR6J`. `tests/test_gr6j.py` checks the GR6J kernel against the independent `hydrogr` implementation, which agrees to within 1e-8 mm/day, and checks that the water balance closes. Settings in `common.py`, each overridable by environment variable:
- `MODEL` (`IWB_MODEL`): `gr6j` or `gr4j`.
- `OBJECTIVE` (`IWB_OBJECTIVE`): `kge_sqrt` or `kge`.
- `PRECIP_SOURCE`: `aorc` or `gridmet`.

Run order: 00 → 03, then 04, 07, 05, 06.

## Key results (GR6J, AORC, KGE on √Q, PET = 0.78 × gridMET ETo)

Earlier configurations are archived in `output/gridmet_run/`, `output/gr4j_aorc_run/` and `output/gr6j_kc1_run/`.

- **Coverage:** 161 IWQIS nitrate sites, of which 74 match a USGS gauge; 63 modelled and 52 calibrated.
- **Annual updating, out-of-sample 2016–2025 hindcast, medians across 50 basins:**

  | | KGE | r | log-NSE | NSE | Bias | Net water import |
  |---|---|---|---|---|---|---|
  | **GR6J, Kc = 0.78 (current)** | **0.73** | **0.83** | **0.67** | **0.57** | +7.7% | **+16 mm/yr** |
  | GR6J, Kc = 1 | 0.73 | 0.81 | 0.59 | 0.56 | +7.5% | +97 mm/yr |
  | GR4J, Kc = 1 | 0.70 | 0.81 | 0.63 | 0.54 | +8.8% | — |

  19 of 50 basins reach KGE ≥ 0.75, and 4 fall below 0.5. The median KGE stays at about 0.73 across these configurations. The crop coefficient mainly improves low flows (p < 1e-9), timing (p = 0.03) and physical plausibility (little imported water).
- **Tests** (`experiments/`):
  - Recalibrating on the previous year only is worse than a fixed calibration; expanding windows are best (`rolling_origin.py`).
  - Precipitation day alignment is correct: timing r is highest with no shift (`improvements.py`).
  - A per-basin calibrated crop coefficient has median 0.75 (IQR 0.73–0.80), confirming the 0.78 used by `iowa-nitrates`.
- **Iowa City cluster** (Clear Creek ×2, Old Mans Creek, English River): runoff fell 15–31% relative to the region after 2016. Published flows agree with USGS field measurements (median ratio 1.00), so the gauges are sound. Local precipitation also fell 3–10% relative to the region, so this is a real dry shift that the model under-responds to.
- **Regionalization, GR6J leave-one-out median KGE:** 0.68 calibrated on the gauge itself, 0.66 attribute-similarity donors, 0.66 spatial proximity, 0.60 ridge regression, 0.59 regional median. Donors are used for the NHDPlus transfer.

## Known issues

- **2021 is still the hardest year** for both models (median next-year KGE −0.09 for GR6J, 0.10 for GR4J). GR6J's exponential store does carry the drought deficit, reaching −300 mm in the Floyd River, but the calibrations also import water through positive exchange. In 16 of 52 basins the calibration switches the store off (`x6` at its 0.1 mm floor).
- **Earlier bias diagnosis (GR4J, fixed split, KGE on Q: about +15%).** This is not a precipitation-product artefact. AORC and gridMET agree to within about 2% in both periods, and both show 2016–25 about 5% wetter than 2002–15. Over the same periods the observed runoff ratio fell (median 0.31 to 0.28), around the 2020–2023 drought. GR4J has no slow groundwater store, so it cannot carry a multi-year storage deficit, which is a known weakness under non-stationary conditions. Daymet, which drove the first 5-basin run, shows a smaller late-period increase; that is why the run had less bias.
- **Low flows too high (resolved).** With Kc = 1 the model imported about +97 mm/yr through exchange, compensating for grass reference ET exceeding crop ET. Kc = 0.78 cuts this to +16 mm/yr.
- **Degree-day melt factor at its upper bound.** `Kf` reaches its 10 mm/°C/day limit in many basins, so the March melt pulse is hard to reproduce with this forcing.
- **Scale transfer.** Parameters come from basins of 12–20,000 km² and are applied to NHDPlus catchments of about 1–5 km². `x4` in particular includes channel routing and should be shorter for local catchments.
- **Water balance terms.** GR4J does not output actual ET, so the plots show `ET + net exchange = P − Q − ΔS`.
- **IWQIS discharge.** IWQIS does not measure discharge itself: 73 of its 74 discharge series are the USGS 00060 record. IWQIS therefore defines which outlets are modelled, and USGS supplies Q.
