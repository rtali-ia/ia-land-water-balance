"""Figures for the Iowa CemaNeige-GR4J/GR6J land-water balance model.

  output/iowa_map.png                  all basins, shaded by validation KGE
  output/iowa_water_balance.png        mean annual P = Q + ET/losses, basins sorted by tile fraction
  output/regionalization_loo.png       leave-one-out skill of each regionalization method
  output/attribute_param_heatmap.png   Spearman rho between basin attributes and parameters
  output/tile_vs_params.png            tile-drained fraction vs parameters / runoff ratio
  output/nhdplus_params_map.png        transferred parameters for every Iowa NHDPlus catchment
  output/annual_update_comparison.png  GR4J vs GR6J, annual updating (out-of-sample hindcast)
  output/dashboards/<site>.png         hydrograph, regime, FDC, annual balance, storages

Usage:  python 06_plot_results.py [--no-dashboards]
"""
import json
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm

from common import MODEL, NITRATES_DIR, OBJECTIVE, OUT_DIR, PRECIP_SOURCE, PROC_DIR, RAW_DIR

# Reference palette (light mode)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
NEUTRAL = "#f0efec"
BLUE = "#2a78d6"      # categorical slot 1: simulated flow / runoff
ORANGE = "#eb6834"    # categorical slot 2: ET + losses
AQUA = "#1baf7a"      # categorical slot 3
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV = ["#0d366b", "#2a78d6", "#9ec5f4", NEUTRAL, "#f4a3a2", "#e34948", "#8f1e1d"]
SEQ = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
DIVC = LinearSegmentedColormap.from_list("div", DIV)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})

PERIOD_COLORS = {"calibration": NEUTRAL, "validation": SURFACE}


def save(fig, name):
    out = OUT_DIR / name
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def load(site):
    sim = pd.read_csv(OUT_DIR / "sims" / f"{site}_sim.csv", parse_dates=["date"], index_col="date")
    attrs = json.loads((PROC_DIR / f"{site}_attrs.json").read_text())
    res = json.loads((OUT_DIR / "params" / f"{site}.json").read_text())
    return sim, attrs, res


def annual_balance(sim):
    """Annual water balance. ET is the model's actual ET when available (GR6J
    output); otherwise the residual P - Q_sim - dS, which then also contains
    the inter-catchment exchange."""
    storage = sim.snow_swe_mm + sim.prod_store_mm + sim.rout_store_mm
    if "exp_store_mm" in sim:
        storage = storage + sim.exp_store_mm
    yr = sim.loc["2002":"2025"]
    a = yr[["precip_mm", "q_sim_mm", "pet_mm"]].resample("YE").sum()
    a["q_obs_mm"] = yr.q_obs_mm.resample("YE").sum(min_count=330)   # NaN if gappy
    s_end = storage.loc["2002":"2025"].resample("YE").last()
    s_start = s_end.shift(1).fillna(storage.loc[:"2001-12-31"].iloc[-1])
    a["dS"] = s_end - s_start
    a["et_losses"] = a.precip_mm - a.q_sim_mm - a.dS
    if "aet_mm" in sim:
        a["et"] = yr.aet_mm.resample("YE").sum()
        a["exchange"] = yr.exchange_mm.resample("YE").sum()
        a["et_label"] = "Actual ET (model)"
    else:
        a["et"] = a.et_losses
        a["et_label"] = "ET + net exchange (P − Q − ΔS)"
    a.index = a.index.year
    return a


# ------------------------------------------------------------- per basin ----
def dashboard(site):
    sim, attrs, res = load(site)
    sc_c, sc_v, p = res["calibration"], res["validation"], res["params"]

    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.15, 1, 1], hspace=0.5, wspace=0.28)
    fig.suptitle(f"{attrs['name']} (USGS {site}), {attrs['drainage_area_km2']:.0f} km²  |  "
                 f"CemaNeige-{MODEL.upper()}, {PRECIP_SOURCE.upper()} precip + gridMET T/PET", x=0.06, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.06, 0.935,
             f"KGE cal {sc_c['KGE']:.2f} / val {sc_v['KGE']:.2f}   ·   NSE cal {sc_c['NSE']:.2f} / "
             f"val {sc_v['NSE']:.2f}   ·   bias val {sc_v['PBIAS']:+.1f}%   ·   split {res['split']}\n"
             f"x1={p['x1']:.0f} mm  x2={p['x2']:.2f} mm/d  x3={p['x3']:.0f} mm  "
             f"x4={p['x4']:.2f} d  "
             + (f"x5={p['x5']:.2f}  x6={p['x6']:.2f} mm  " if "x5" in p else "")
             + f"CTG={p['CTG']:.2f}  Kf={p['Kf']:.2f} mm/°C/d",
             color=INK2, fontsize=9.5, va="top")

    # (a) daily hydrograph, full record, calibration period shaded
    ax = fig.add_subplot(gs[0, :])
    c = sim.period == "calibration"
    if c.any():
        ax.axvspan(sim.index[c][0], sim.index[c][-1], color=NEUTRAL, lw=0)
    ax.plot(sim.index, sim.q_obs_mm, color=INK, lw=0.8, label="Observed (USGS)")
    ax.plot(sim.index, sim.q_sim_mm, color=BLUE, lw=0.9, alpha=0.9, label="Simulated")
    ax.set_yscale("log")
    qo = sim.q_obs_mm.dropna()
    ax.set_ylim(max(0.002, qo.quantile(0.001) / 2), qo.max() * 1.6)
    ax.set_ylabel("Streamflow [mm/day]")
    ax.set_title("Daily streamflow (log scale); shaded = calibration period")
    ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(1, 1.0))
    ax.margins(x=0)

    scored = sim[sim.period != "warmup"]

    # (b) mean monthly regime (days with observations only)
    ax = fig.add_subplot(gs[1, 0])
    s = scored.dropna(subset=["q_obs_mm"])
    m = s.groupby(s.index.month)[["q_obs_mm", "q_sim_mm"]].mean()
    ax.plot(m.index, m.q_obs_mm, color=INK, lw=2, marker="o", ms=5, label="Observed")
    ax.plot(m.index, m.q_sim_mm, color=BLUE, lw=2, marker="o", ms=5, label="Simulated")
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_ylabel("Mean flow [mm/day]")
    ax.set_title("Seasonal regime")
    ax.set_ylim(bottom=0)
    ax.legend()

    # (c) flow duration curve
    ax = fig.add_subplot(gs[1, 1])
    for col, colr, lbl in [("q_obs_mm", INK, "Observed"), ("q_sim_mm", BLUE, "Simulated")]:
        q = np.sort(s[col].values)[::-1]
        ax.plot(100 * np.arange(1, len(q) + 1) / (len(q) + 1), q, color=colr, lw=2, label=lbl)
    ax.set_yscale("log")
    ax.set_xlabel("Exceedance probability [%]")
    ax.set_ylabel("Streamflow [mm/day]")
    ax.set_title("Flow duration curve")
    ax.legend()

    # (d) observed vs simulated monthly volumes
    ax = fig.add_subplot(gs[1, 2])
    mm = s[["q_obs_mm", "q_sim_mm"]].resample("ME").sum(min_count=25).dropna()
    lim = mm.max().max() * 1.05
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--")
    ax.scatter(mm.q_obs_mm, mm.q_sim_mm, s=16, color=BLUE, alpha=0.6,
               edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Observed [mm/month]"); ax.set_ylabel("Simulated [mm/month]")
    ax.set_title("Monthly runoff volumes")

    # (e) annual water balance
    ax = fig.add_subplot(gs[2, :2])
    a = annual_balance(sim)
    x = a.index.values
    ax.bar(x, a.q_sim_mm, width=0.75, color=BLUE, label="Runoff Q (sim)",
           edgecolor=SURFACE, linewidth=1)
    ax.bar(x, a.et, bottom=a.q_sim_mm, width=0.75, color=ORANGE,
           label=a.et_label.iloc[0], edgecolor=SURFACE, linewidth=1)
    ax.scatter(x, a.precip_mm, color=INK, s=22, zorder=3, label="Precipitation P")
    ax.scatter(x, a.q_obs_mm, color=SURFACE, edgecolor=INK, s=22, zorder=3,
               linewidth=1.2, label="Runoff Q (obs)")
    ax.set_ylabel("mm/year")
    title = "Annual land-water balance"
    if "exchange" in a:
        title += f"  (mean net exchange {a.exchange.mean():+.0f} mm/yr; gap to P = ΔS − exchange)"
    ax.set_title(title, fontsize=10)
    ax.legend(ncol=4, loc="upper left", fontsize=8.5)
    ax.set_ylim(0, a.precip_mm.max() * 1.2)
    ax.set_xlim(x.min() - 0.6, x.max() + 0.6)

    # (f) storages (small multiples, one axis each), spanning the 2020-23 drought
    has_exp = "exp_store_mm" in sim
    sub = gs[2, 2].subgridspec(3 if has_exp else 2, 1, hspace=0.2)
    win = sim.loc["2019-10-01":"2023-09-30"]
    ax1 = fig.add_subplot(sub[0])
    ax1.fill_between(win.index, win.snow_swe_mm, color=BLUE, alpha=0.35, lw=0)
    ax1.plot(win.index, win.snow_swe_mm, color=BLUE, lw=1.2)
    ax1.set_ylabel("SWE\n[mm]")
    ax1.set_title("Model storages, WY2020-2023")
    ax1.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(sub[1], sharex=ax1)
    ax2.plot(win.index, win.prod_store_mm / p["x1"] * 100, color=INK2, lw=1.2)
    ax2.set_ylabel("Soil\n[% x1]")
    ax2.set_ylim(0, 100)
    last = ax2
    if has_exp:
        ax2.tick_params(labelbottom=False)
        ax3 = fig.add_subplot(sub[2], sharex=ax1)
        ax3.axhline(0, color=MUTED, lw=0.8)
        ax3.plot(win.index, win.exp_store_mm, color=AQUA, lw=1.2)
        ax3.set_ylabel("Exp. store\n[mm]")
        last = ax3
    last.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    last.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    save(fig, f"dashboards/{site}.png")


# ------------------------------------------------------------ statewide -----
def iowa_outline():
    return gpd.read_file(RAW_DIR / "iowa_boundary.geojson").to_crs(4326)


def style_map(ax, iowa):
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    xmin, ymin, xmax, ymax = iowa.total_bounds
    ax.set_xlim(xmin - 0.9, xmax + 0.3)
    ax.set_ylim(ymin - 0.2, ymax + 0.9)
    ax.set_aspect(1 / np.cos(np.radians(42)))


def iowa_map(summ):
    iowa = iowa_outline()
    ok = summ[summ.status == "ok"].copy()
    basins = []
    for s in ok.index:
        g = gpd.read_file(RAW_DIR / s / "basin.geojson").to_crs(4326).union_all()
        basins.append({"site": s, "geometry": g, "kge": ok.loc[s, "KGE_val"]})
    gdf = gpd.GeoDataFrame(basins, crs=4326)
    gdf["area"] = gdf.to_crs(5070).area
    gdf = gdf.sort_values("area", ascending=False)       # small basins drawn on top

    norm = Normalize(vmin=0.2, vmax=1.0)
    fig, ax = plt.subplots(figsize=(11, 7.5))
    iowa.plot(ax=ax, facecolor=NEUTRAL, edgecolor=INK2, lw=1.2)
    gdf.plot(ax=ax, color=[SEQ(norm(k)) for k in gdf.kge], edgecolor=SURFACE, lw=0.8)
    for s in ok.index:
        a = json.loads((PROC_DIR / f"{s}_attrs.json").read_text())
        ax.plot(a["gauge_lon"], a["gauge_lat"], "^", ms=6, color=INK, mec=SURFACE, mew=0.8, zorder=4)
    style_map(ax, iowa)
    ax.set_title(f"CemaNeige-{MODEL.upper()} at {len(ok)} IWQIS / USGS gauges, shaded by validation KGE "
                 f"(median {ok.KGE_val.median():.2f})")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=SEQ), ax=ax, shrink=0.55, pad=0.01)
    cb.set_label("KGE (validation)"); cb.outline.set_visible(False)
    ax.plot([], [], "^", color=INK, ms=7, label="USGS gauge with IWQIS nitrate sensor")
    ax.legend(loc="lower left")
    save(fig, "iowa_map.png")


def iowa_water_balance(summ, att):
    ok = [s for s in summ[summ.status == "ok"].index if s in att.index]
    rows = []
    for s in ok:
        ab = annual_balance(load(s)[0])
        rows.append({"site": s, "name": att.loc[s, "name"], "tile": att.loc[s, "tile_frac"],
                     "et_label": ab.et_label.iloc[0], **ab.mean(numeric_only=True).to_dict()})
    d = pd.DataFrame(rows).sort_values("tile")
    fig, ax = plt.subplots(figsize=(10, 0.24 * len(d) + 1.8))
    y = np.arange(len(d))
    ax.barh(y, d.q_sim_mm, color=BLUE, height=0.75, label="Runoff Q (sim)",
            edgecolor=SURFACE, lw=1)
    ax.barh(y, d.et, left=d.q_sim_mm, color=ORANGE, height=0.75,
            label=d.et_label.iloc[0], edgecolor=SURFACE, lw=1)
    ax.scatter(d.precip_mm, y, color=INK, s=18, zorder=3, label="Precipitation P")
    ax.scatter(d.q_obs_mm, y, color=SURFACE, edgecolor=INK, s=18, lw=1.1, zorder=3,
               label="Runoff Q (obs)")
    ax.set_yticks(y, [f"{n.replace(', IA', '')[:34]}  ({t:.0%} tiled)"
                      for n, t in zip(d.name, d.tile)], fontsize=7.5)
    ax.set_xlabel("mm/year (2002-2025 mean)")
    ax.set_title("Mean annual land-water balance, basins sorted by tile-drained fraction")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.01)
    save(fig, "iowa_water_balance.png")


def regionalization_plot():
    loo = pd.read_csv(OUT_DIR / "regionalization_loo.csv", dtype={"site": str})
    methods = [("calibrated", "Calibrated\n(own gauge)"), ("regional", "Regional\nmedian"),
               ("regression", "Ridge\nregression"), ("similarity", "Attribute\nsimilarity"),
               ("proximity", "Spatial\nproximity")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rng = np.random.default_rng(0)
    for i, (m, _) in enumerate(methods):
        v = loo[m].clip(lower=-0.5)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=BLUE if i else INK,
                   alpha=0.55, edgecolor=SURFACE, lw=0.4)
        med = loo[m].median()
        ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2.2)
        ax.text(i + 0.33, med, f"{med:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(methods)), [l for _, l in methods])
    ax.set_ylabel("KGE on validation period")
    ax.set_ylim(-0.55, 1)
    ax.set_title("Predicting ungauged basins: leave-one-out skill (bar = median; values < −0.5 clipped)")
    save(fig, "regionalization_loo.png")


def heatmap():
    corr = pd.read_csv(OUT_DIR / "attribute_param_correlations.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.grid(False)
    im = ax.imshow(corr.values, cmap=DIVC, norm=TwoSlopeNorm(0, -0.8, 0.8), aspect="auto")
    ax.set_xticks(range(corr.shape[1]), corr.columns)
    ax.set_yticks(range(corr.shape[0]), corr.index)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.5 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Spearman ρ"); cb.outline.set_visible(False)
    ax.set_title("Basin attributes vs calibrated parameters")
    save(fig, "attribute_param_heatmap.png")


def tile_scatter(summ, att):
    ok = [s for s in summ[summ.status == "ok"].index if s in att.index]
    d = summ.loc[ok].join(att[["tile_frac", "obs_runoff_ratio", "bfi"]])
    panels = [("x1", "x1 production store [mm]", True), ("x3", "x3 routing store [mm]", True),
              ("x4", "x4 unit hydrograph [days]", False), ("x2", "x2 exchange [mm/day]", False),
              ("obs_runoff_ratio", "Observed runoff ratio Q/P", False),
              ("bfi", "Baseflow index (StreamCat)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (col, lbl, logy) in zip(axes.ravel(), panels):
        ax.scatter(d.tile_frac * 100, d[col], s=24, color=BLUE, alpha=0.75,
                   edgecolor=SURFACE, lw=0.6)
        if logy:
            ax.set_yscale("log")
        rho = d[["tile_frac", col]].corr("spearman").iloc[0, 1]
        ax.set_title(f"{lbl}   (ρ = {rho:+.2f})", fontsize=10)
        ax.set_xlabel("Tile-drained area [%]")
    fig.suptitle("Tile drainage vs calibrated model behaviour", x=0.06, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "tile_vs_params.png")


def nhdplus_map():
    f = OUT_DIR / "nhdplus_params.parquet"
    if not f.exists():
        return
    prm = pd.read_parquet(f)
    cats = gpd.read_parquet(NITRATES_DIR / "data" / "raw" / "nhdplus" / "catchments.parquet")
    cats = cats.rename(columns={"featureid": "comid"}).set_index("comid")
    cov = pd.read_parquet(NITRATES_DIR / "data" / "processed" / "covariates_static.parquet")
    g = cats[["geometry"]].join(prm, how="inner").join(cov.set_index("comid")[["tile_drain_frac"]])
    iowa = iowa_outline()
    panels = [("tile_drain_frac", "Tile-drained fraction", 0, 1),
              ("x1", "x1 production store [mm]", *np.nanpercentile(g.x1, [2, 98])),
              ("x3", "x3 routing store [mm]", *np.nanpercentile(g.x3, [2, 98])),
              ("x4", "x4 unit hydrograph [days]", *np.nanpercentile(g.x4, [2, 98]))]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    for ax, (col, lbl, lo, hi) in zip(axes.ravel(), panels):
        g.plot(ax=ax, column=col, cmap=SEQ, vmin=lo, vmax=hi, lw=0, rasterized=True)
        iowa.boundary.plot(ax=ax, color=INK2, lw=0.8)
        style_map(ax, iowa)
        ax.set_xlim(iowa.total_bounds[0] - 0.1, iowa.total_bounds[2] + 0.1)
        ax.set_ylim(iowa.total_bounds[1] - 0.1, iowa.total_bounds[3] + 0.1)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=SEQ), ax=ax,
                          shrink=0.7, pad=0.01)
        cb.outline.set_visible(False)
        ax.set_title(lbl)
    fig.suptitle(f"Parameters transferred to {len(g):,} NHDPlus catchments "
                 f"(method: {prm.method.iloc[0]})", x=0.06, ha="left", fontsize=13,
                 fontweight="bold")
    save(fig, "nhdplus_params_map.png")


def annual_update_plot():
    """Out-of-sample hindcast skill under annual updating, GR4J vs GR6J."""
    base = OUT_DIR / "annual"
    models = [m for m in ("gr4j", "gr6j") if (base / m / "hindcast_summary.csv").exists()]
    if not models:
        return
    colors = {"gr4j": INK, "gr6j": BLUE}
    summ = {m: pd.read_csv(base / m / "hindcast_summary.csv", dtype={"site": str})
            .query("status == 'ok'").set_index("site") for m in models}
    common_sites = sorted(set.intersection(*[set(v.index) for v in summ.values()]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 1, 1.6]})
    rng = np.random.default_rng(0)
    for ax, col, lbl in [(axes[0], "KGE_hindcast", "KGE, 2016-2025 hindcast"),
                         (axes[1], "logNSE_hindcast", "log-NSE (low flows)")]:
        for i, m in enumerate(models):
            v = summ[m].loc[common_sites, col].clip(lower=-1)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, alpha=0.5,
                       color=colors[m], lw=0)
            med = summ[m].loc[common_sites, col].median()
            ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=2.2)
            ax.text(i + 0.33, med, f"{med:.2f}", va="center", fontsize=9)
        ax.set_xticks(range(len(models)), [m.upper() for m in models])
        ax.set_title(lbl, fontsize=10.5)
        ax.set_ylim(-1.05, 1)
    ax = axes[2]
    for m in models:
        by = pd.read_csv(base / m / "hindcast_by_year.csv", dtype={"site": str})
        by = by[by.site.isin(common_sites)]
        med = by.groupby("target_year").KGE_pred.median()
        ax.plot(med.index, med.values, marker="o", ms=5, lw=2, color=colors[m], label=m.upper())
    ax.set_xticks(range(2016, 2026))
    ax.set_ylabel("Median KGE across basins")
    ax.set_title("Next-year prediction skill by year", fontsize=10.5)
    ax.legend()
    fig.suptitle(f"Annual updating (calibrate on 2002..Y−1, predict Y), {len(common_sites)} basins, "
                 f"objective {OBJECTIVE}", x=0.06, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "annual_update_comparison.png")


def main():
    summ = pd.read_csv(OUT_DIR / "calibration_summary.csv", dtype={"site": str}).set_index("site")
    att = pd.read_csv(PROC_DIR / "basin_attributes.csv", dtype={"site": str}).set_index("site")
    iowa_map(summ)
    iowa_water_balance(summ, att)
    tile_scatter(summ, att)
    if (OUT_DIR / "regionalization_loo.csv").exists():
        regionalization_plot()
        heatmap()
        nhdplus_map()
    annual_update_plot()
    if "--no-dashboards" not in sys.argv:
        for s in summ[summ.status == "ok"].index:
            dashboard(s)


if __name__ == "__main__":
    main()
