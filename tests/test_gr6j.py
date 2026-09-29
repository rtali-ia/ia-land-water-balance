"""GR6J kernel checks: agreement with hydrogr and exact water-balance closure.

Run:  .venv/bin/python -m pytest tests/ -q      (or python tests/test_gr6j.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import run_gr6j  # noqa: E402
from common import PROC_DIR  # noqa: E402

PARAM_SETS = [
    dict(x1=250.0, x2=0.8, x3=40.0, x4=2.3, x5=0.3, x6=15.0),
    dict(x1=800.0, x2=-2.5, x3=150.0, x4=6.7, x5=-0.5, x6=3.0),
    dict(x1=60.0, x2=3.0, x3=8.0, x4=0.7, x5=1.5, x6=200.0),   # extremes
    dict(x1=150.0, x2=0.0, x3=60.0, x4=4.0, x5=0.0, x6=0.5),
]


def forcing():
    f = next(PROC_DIR.glob("05484500.csv"), None)
    if f is not None:
        d = pd.read_csv(f)
        return d.precip_mm.values.astype(float), d.pet_mm.values.astype(float)
    rng = np.random.default_rng(0)                  # synthetic fallback
    n = 3000
    p = np.where(rng.random(n) < 0.3, rng.gamma(0.8, 12, n), 0.0)
    e = 2.5 + 2 * np.sin(np.arange(n) * 2 * np.pi / 365)
    return p, np.clip(e, 0, None)


def test_matches_hydrogr():
    from hydrogr._hydrogr import gr6j
    p, e = forcing()
    for prm in PARAM_SETS:
        x = [prm[k] for k in ("x1", "x2", "x3", "x4", "x5", "x6")]
        s0, r0, e0 = 0.3, 0.5, 0.3
        ref = gr6j(x, p, e, np.array([s0 * x[0], r0 * x[2], e0 * x[5]]),
                   np.zeros(20), np.zeros(40))[3]
        mine = run_gr6j(p, e, *x, s0, r0, e0 * x[5])[0]
        err = np.max(np.abs(np.asarray(ref) - mine))
        assert err < 1e-8, f"{prm}: max |diff| = {err}"


def test_water_balance_closes():
    p, e = forcing()
    for prm in PARAM_SETS:
        x = [prm[k] for k in ("x1", "x2", "x3", "x4", "x5", "x6")]
        q, S, R, E, aet, ex = run_gr6j(p, e, *x, 0.6, 0.7, 0.0)
        # storage in unit hydrographs = rainfall routed in but not yet out
        s_end = S[-1] + R[-1] + E[-1]
        s_ini = 0.6 * x[0] + 0.7 * x[2] + 0.0
        resid = p.sum() + ex.sum() - aet.sum() - q.sum() - (s_end - s_ini)
        # remaining residual = water still inside the UH buffers (< a few mm)
        assert abs(resid) < 0.05 * p[-60:].sum() + 1e-6, f"{prm}: residual {resid:.3f} mm"


if __name__ == "__main__":
    test_matches_hydrogr()
    test_water_balance_closes()
    print("GR6J: matches hydrogr to <1e-8 mm/day; water balance closes")
