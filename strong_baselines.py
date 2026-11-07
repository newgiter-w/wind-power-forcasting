"""P2: shared-lead quantile GBM (R1) and R1 + weather-derived features (R2).

R1: unfold training days to (origin,h), ONE q model serves h=1..24; features =
R0's 107-dim B_speed (base 35 + full U100/V100/ws100 curve 72) + lead h.
R2: R1 + per-lead explicit U/V/ws/ws^3/dir-sin-cos/hour-calendar +-1,2h and diffs.
Both use the same q05/q95, train/validation split, rolling-84 CQR, clip[0,1].
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from gefcom2014_experiment import metrics
from gefcom_protocol_v2 import PKG, index, prepare
from online_calibration import simulate

ROOT = Path(__file__).parent
OUT = ROOT / 'out_gefcom_strong_baselines_v1'


def base_features(history, origin):
    x = np.arange(24)
    o = pd.Timestamp(origin)
    stats = [history.mean(), history.std(), history[-1], history.min(), history.max(),
             np.polyfit(x, history, 1)[0], np.sum(np.abs(np.diff(history)) > .1)]
    cal = [np.sin(2 * np.pi * o.dayofyear / 365.25), np.cos(2 * np.pi * o.dayofyear / 365.25),
           np.sin(2 * np.pi * o.hour / 24), np.cos(2 * np.pi * o.hour / 24)]
    return np.r_[history, stats, cal]


def _shift(a, k):
    out = np.full_like(a, np.nan)
    if k > 0:
        out[:, k:] = a[:, :-k]
    elif k < 0:
        out[:, :k] = a[:, -k:]
    else:
        out[:] = a
    return out


def order_quantile_bounds(q05, q95):
    """Sort both endpoints using the unmodified heads, including crossed cells."""
    return np.minimum(q05, q95), np.maximum(q05, q95)


def build_shared(farm, method):
    S = len(farm['history'])
    base = np.array([base_features(h, o) for h, o in zip(farm['history'], farm['origins'])])
    w = farm['nwp']
    u100 = w[:, :, 2]; v100 = w[:, :, 3]; ws100 = np.hypot(u100, v100)
    curve = np.stack([u100, v100, ws100], axis=2).reshape(S, -1)          # (S,72)
    base_rep = np.repeat(base, 24, axis=0)                                # (S*24,35)
    curve_rep = np.repeat(curve, 24, axis=0)                              # (S*24,72)
    lead = np.tile(np.arange(1, 25), S).astype(float)                     # (S*24,)
    X = np.c_[base_rep, curve_rep, lead]
    if method == 'R2':
        ws = ws100; ws3 = ws ** 3
        safe = np.maximum(ws, 1e-6)
        dir_sin = np.where(ws > 1e-6, v100 / safe, 0.0)
        dir_cos = np.where(ws > 1e-6, u100 / safe, 0.0)
        h_idx = np.arange(1, 25)
        h_sin = np.broadcast_to(np.sin(2 * np.pi * h_idx / 24)[None, :], ws.shape)
        h_cos = np.broadcast_to(np.cos(2 * np.pi * h_idx / 24)[None, :], ws.shape)
        u_p1 = _shift(u100, 1); u_p2 = _shift(u100, 2); u_n1 = _shift(u100, -1); u_n2 = _shift(u100, -2)
        du_fwd = u_n1 - u100; du_bwd = u100 - u_p1
        extra = np.stack([u100, v100, ws, ws3, dir_sin, dir_cos, h_sin, h_cos,
                          u_p1, u_p2, u_n1, u_n2, du_fwd, du_bwd], axis=2)  # (S,24,14)
        X = np.c_[X, extra.reshape(S * 24, 14)]
    y = farm['y'].reshape(S * 24)
    return X, y, S


def fit_predict(X, y, train_mask_unfolded, q):
    scaler = StandardScaler().fit(X[train_mask_unfolded])
    xs = scaler.transform(X)
    m = HistGradientBoostingRegressor(loss='quantile', quantile=q, **PKG)
    m.fit(xs[train_mask_unfolded], y[train_mask_unfolded])
    return m.predict(xs).reshape(-1, 24), m, scaler


def run_validation(farms, methods):
    rows = []
    for farm, d in farms.items():
        tr = index(d, 'train', True); vs = index(d, 'validation_seed'); vi = index(d, 'validation')
        tr_unfolded = np.repeat(np.isin(np.arange(len(d['origins'])), tr), 24)
        for method in methods:
            X, y, S = build_shared(d, method)
            assert X.shape[0] == S * 24
            lo, m_lo, sc_lo = fit_predict(X, y, tr_unfolded, 0.05)
            hi, m_hi, sc_hi = fit_predict(X, y, tr_unfolded, 0.95)
            lo, hi = order_quantile_bounds(lo, hi)
            res = simulate(d['origins'], d['ends'], d['y'], lo, hi, d['valid'], vs, vi, 'rolling', 84)
            L = np.clip(res['lo'], 0, 1); U = np.clip(res['hi'], 0, 1)
            met = metrics(d['y'][vi], L, U, d['valid'][vi], d['history'][vi, -1])
            for g, v in met.items():
                rows.append(dict(farm=farm, method=method, group=g, **v))
            print(f'  farm{farm:02d} {method} val done', flush=True)
    df = pd.DataFrame(rows)
    macro = df.groupby(['method', 'group'])[['coverage', 'width', 'IS', 'lower_miss', 'upper_miss']].mean().reset_index()
    df.to_csv(OUT / 'p2_validation_per_farm.csv', index=False)
    macro.to_csv(OUT / 'p2_validation_macro.csv', index=False)
    return macro


def main():
    OUT.mkdir(exist_ok=True)
    began = time.perf_counter()
    farms, _ = prepare(2014)
    macro = run_validation(farms, ['R1', 'R2'])
    print('\n===== P2 validation (2014 Apr-Jun, clipped CQR) =====')
    for g in ('all', 'up_ramp', 'near_zero'):
        line = f'  {g:9s}'
        for m in ('R1', 'R2'):
            r = macro[(macro.method == m) & (macro.group == g)].iloc[0]
            line += f"  {m}: cov={r['coverage']*100:5.1f}% w={r['width']*100:5.2f}% IS={r['IS']:.4f}"
        print(line)
    print(f'\n[done] ({time.perf_counter()-began:.1f}s)')


if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
