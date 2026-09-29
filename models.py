"""Rainfall-runoff models behind one interface: CemaNeige snow + GR4J or GR6J.

  gr4j   rrmpg's CemaNeige-GR4J kernel (Perrin et al., 2003)
  gr6j   CemaNeige (rrmpg kernel) + GR6J written here in numba, following
         airGR's frun_GR6J (Pushpalatha et al., 2011, J. Hydrol. 411:66-76).
         Verified against the independent hydrogr implementation
         (tests/test_gr6j.py).

GR6J differs from GR4J in two ways that matter for Iowa:
  * inter-catchment exchange depends on routing-store level:
        F = x2 * (R / x3 - x5)
  * 40% of the UH1 output feeds an *exponential store* that has no lower
    bound. It can hold a large negative level after a drought, i.e. it keeps
    a multi-year deficit that GR4J forgets within months. x6 [mm] sets how
    quickly it drains.

Usage:
    inputs = model_inputs(df, attrs)          # forcing arrays, prepared once
    q = simulate(inputs, params)              # daily streamflow [mm/day]
    out = simulate(inputs, params, return_states=True)   # dict of all fluxes/states
"""
import numpy as np
from numba import njit

from rrmpg.models.cemaneige_model import run_cemaneige
from rrmpg.models.cemaneige_utils import calculate_solid_fraction
from rrmpg.models.cemaneigegr4j_model import run_cemaneigegr4j

from common import MODEL, PET_CROP_COEF

PET_COLUMN = "pet_mm"        # gridMET ASCE grass reference ET ("pet_oudin_mm" = Oudin),
                             # scaled by PET_CROP_COEF in model_inputs()

# Initial states. Snow 0 mm (Jan 1); stores as fractions of capacity.
# The exponential store starts empty (0 mm) and fills during the warm-up.
INIT = dict(snow_pack_init=0.0, thermal_state_init=0.0, s_init=0.6, r_init=0.7, exp_init=0.0)

SNOW_BOUNDS = {
    "CTG": (0.0, 1.0),      # snowpack thermal-state weighting [-]
    "Kf": (0.0, 10.0),      # degree-day melt factor [mm/°C/day]
}
GR4J_BOUNDS = {
    "x1": (50.0, 3000.0),   # production store capacity [mm]
    "x2": (-10.0, 5.0),     # groundwater exchange coefficient [mm/day]
    "x3": (5.0, 1000.0),    # routing store capacity [mm]
    "x4": (0.5, 10.0),      # unit hydrograph time base [days]
}
GR6J_BOUNDS = {
    **GR4J_BOUNDS,
    "x5": (-1.0, 2.0),      # exchange threshold, fraction of x3 [-]
    "x6": (0.1, 500.0),     # exponential store coefficient [mm]
}
BOUNDS = {**SNOW_BOUNDS, **(GR6J_BOUNDS if MODEL == "gr6j" else GR4J_BOUNDS)}
PARAMS = list(BOUNDS)
LOG_PARAMS = {"x1", "x3", "x4", "x6"}       # positive, scale-like parameters

_SNOW_DTYPE = np.dtype([("CTG", np.float64), ("Kf", np.float64)])
_GR4J_DTYPE = np.dtype([(k, np.float64) for k in ["CTG", "Kf", "x1", "x2", "x3", "x4"]])


# ------------------------------------------------------------------- inputs --
def model_inputs(df, attrs):
    """Single-layer CemaNeige inputs (prepared as rrmpg does) plus PET
    (reference ET x crop coefficient)."""
    z = np.array([float(attrs["ref_elev_m"])])
    prec = df.precip_mm.values[:, None].astype(np.float64)
    tmean = df.temp_c.values[:, None].astype(np.float64)
    tmin = df.tmin_c.values[:, None].astype(np.float64)
    tmax = df.tmax_c.values[:, None].astype(np.float64)
    frac_solid = calculate_solid_fraction(prec, z, tmean, tmin, tmax)
    return prec, tmean, frac_solid, df[PET_COLUMN].values.astype(np.float64) * PET_CROP_COEF


# ------------------------------------------------------------------- GR6J ---
@njit(cache=True)
def _ss1(t, x4):
    if t <= 0.0:
        return 0.0
    if t < x4:
        return (t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def _ss2(t, x4):
    if t <= 0.0:
        return 0.0
    if t <= x4:
        return 0.5 * (t / x4) ** 2.5
    if t < 2.0 * x4:
        return 1.0 - 0.5 * (2.0 - t / x4) ** 2.5
    return 1.0


@njit(cache=True)
def run_gr6j(prec, etp, x1, x2, x3, x4, x5, x6, s_init, r_init, e_init):
    """GR6J daily kernel (airGR frun_GR6J). s_init, r_init are fractions of
    x1, x3; e_init is the exponential-store level in mm.

    Returns q, production store, routing store, exponential store,
    actual ET, and actual net exchange (+ = gain) as daily arrays.
    """
    n = len(prec)
    q = np.zeros(n)
    s_st = np.zeros(n)
    r_st = np.zeros(n)
    e_st = np.zeros(n)
    ae = np.zeros(n)
    exch_act = np.zeros(n)

    nh1 = int(np.ceil(x4))
    nh2 = int(np.ceil(2.0 * x4))
    o1 = np.zeros(nh1)
    o2 = np.zeros(nh2)
    for j in range(nh1):
        o1[j] = _ss1(j + 1.0, x4) - _ss1(float(j), x4)
    for j in range(nh2):
        o2[j] = _ss2(j + 1.0, x4) - _ss2(float(j), x4)
    uh1 = np.zeros(nh1)
    uh2 = np.zeros(nh2)

    S = s_init * x1
    R = r_init * x3
    E = e_init
    for t in range(n):
        p, ep = prec[t], etp[t]
        # interception and production store
        if p >= ep:
            pn = p - ep
            ws = min(pn / x1, 13.0)
            tws = np.tanh(ws)
            ps = x1 * (1.0 - (S / x1) ** 2) * tws / (1.0 + S / x1 * tws)
            pr = pn - ps
            S += ps
            aet = ep
        else:
            en = ep - p
            ws = min(en / x1, 13.0)
            tws = np.tanh(ws)
            er = S * (2.0 - S / x1) * tws / (1.0 + (1.0 - S / x1) * tws)
            S -= er
            pr = 0.0
            aet = er + p
        if S < 0.0:
            S = 0.0
        # percolation
        perc = S * (1.0 - (1.0 + (S / (2.25 * x1)) ** 4) ** -0.25)
        S -= perc
        pr += perc
        # unit hydrographs: 90% -> UH1, 10% -> UH2
        for j in range(nh1 - 1):
            uh1[j] = uh1[j + 1] + o1[j] * 0.9 * pr
        uh1[nh1 - 1] = o1[nh1 - 1] * 0.9 * pr
        for j in range(nh2 - 1):
            uh2[j] = uh2[j + 1] + o2[j] * 0.1 * pr
        uh2[nh2 - 1] = o2[nh2 - 1] * 0.1 * pr
        q9 = uh1[0]
        q1 = uh2[0]
        # potential inter-catchment exchange
        F = x2 * (R / x3 - x5)
        # routing store (60% of UH1)
        a1 = F
        if R + 0.6 * q9 + F < 0.0:
            a1 = -R - 0.6 * q9
        R = R + 0.6 * q9 + F
        if R < 0.0:
            R = 0.0
        qr = R * (1.0 - (1.0 + (R / x3) ** 4) ** -0.25)
        R -= qr
        # exponential store (40% of UH1); unbounded below
        E = E + 0.4 * q9 + F
        ar = min(max(E / x6, -33.0), 33.0)
        if ar > 7.0:
            qre = E + x6 / np.exp(ar)
        elif ar < -7.0:
            qre = x6 * np.exp(ar)
        else:
            qre = x6 * np.log(np.exp(ar) + 1.0)
        E -= qre
        # direct branch (UH2)
        a2 = F
        if q1 + F < 0.0:
            a2 = -q1
        qd = max(0.0, q1 + F)

        q[t] = qr + qd + qre
        s_st[t] = S
        r_st[t] = R
        e_st[t] = E
        ae[t] = aet
        exch_act[t] = a1 + a2 + F
    return q, s_st, r_st, e_st, ae, exch_act


# --------------------------------------------------------------- dispatch ---
def simulate(inputs, p, return_states=False):
    """Run CemaNeige + the configured GR model for one parameter dict."""
    prec, tmean, frac_solid, etp = inputs
    if MODEL == "gr4j":
        arr = np.zeros(1, dtype=_GR4J_DTYPE)
        for k in _GR4J_DTYPE.names:
            arr[k] = p[k]
        q, G, eTG, s_st, r_st = run_cemaneigegr4j(
            prec, tmean, etp, frac_solid, INIT["snow_pack_init"],
            INIT["thermal_state_init"], INIT["s_init"], INIT["r_init"], arr[0])
        if not return_states:
            return q
        return {"q": q, "snow": G[:, 0], "prod": s_st, "rout": r_st}

    snow = np.zeros(1, dtype=_SNOW_DTYPE)
    snow["CTG"], snow["Kf"] = p["CTG"], p["Kf"]
    liquid, G, eTG = run_cemaneige(prec, tmean, frac_solid, INIT["snow_pack_init"],
                                   INIT["thermal_state_init"], snow[0])
    q, s_st, r_st, e_st, ae, ex = run_gr6j(
        liquid, etp, p["x1"], p["x2"], p["x3"], p["x4"], p["x5"], p["x6"],
        INIT["s_init"], INIT["r_init"], INIT["exp_init"])
    if not return_states:
        return q
    return {"q": q, "snow": G[:, 0], "prod": s_st, "rout": r_st, "exp": e_st,
            "aet": ae, "exchange": ex, "liquid": liquid}
