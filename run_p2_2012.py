"""P2 evaluation on GEFCom2012 (frozen R2, single wind height + nwp_age)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from gefcom2014_experiment import metrics
from gefcom_protocol_v2 import index, prepare
from online_calibration import simulate
from strong_baselines import base_features, fit_predict, _shift, order_quantile_bounds

ROOT = Path(__file__).parent
OUT = ROOT / 'out_gefcom_strong_baselines_v1'
SRC = ROOT / 'out_gefcom_protocol_v2' / 'gefcom2012'


def build_shared_2012(farm, method):
    S = len(farm['history'])
    base = np.array([base_features(h, o) for h, o in zip(farm['history'], farm['origins'])])
    w = farm['nwp']  # (S, 24, 2) u, v
    u = w[:, :, 0]; v = w[:, :, 1]; ws = np.hypot(u, v)
    curve = np.stack([u, v, ws], axis=2).reshape(S, -1)          # (S,72)
    age = farm['meta'].nwp_age_hours.to_numpy()[:, None] / 48.    # (S,1)
    base_rep = np.repeat(base, 24, axis=0)
    curve_rep = np.repeat(curve, 24, axis=0)
    age_rep = np.repeat(age, 24, axis=0)
    lead = np.tile(np.arange(1, 25), S).astype(float)
    X = np.c_[base_rep, curve_rep, age_rep, lead]
    if method == 'R2':
        ws3 = ws ** 3
        safe = np.maximum(ws, 1e-6)
        dir_sin = np.where(ws > 1e-6, v / safe, 0.0)
        dir_cos = np.where(ws > 1e-6, u / safe, 0.0)
        h_idx = np.arange(1, 25)
        h_sin = np.broadcast_to(np.sin(2 * np.pi * h_idx / 24)[None, :], ws.shape)
        h_cos = np.broadcast_to(np.cos(2 * np.pi * h_idx / 24)[None, :], ws.shape)
        u_p1 = _shift(u, 1); u_p2 = _shift(u, 2); u_n1 = _shift(u, -1); u_n2 = _shift(u, -2)
        du_fwd = u_n1 - u; du_bwd = u - u_p1
        extra = np.stack([u, v, ws, ws3, dir_sin, dir_cos, h_sin, h_cos,
                          u_p1, u_p2, u_n1, u_n2, du_fwd, du_bwd], axis=2)
        X = np.c_[X, extra.reshape(S * 24, 14)]
    y = farm['y'].reshape(S * 24)
    return X, y, S


def main():
    OUT.mkdir(exist_ok=True)
    began = time.perf_counter()
    farms, _ = prepare(2012)
    rows = []
    for farm, d in farms.items():
        tr = index(d, 'train', True); es = index(d, 'validation'); ei = index(d, 'evaluation')
        tr_unfolded = np.repeat(np.isin(np.arange(len(d['origins'])), tr), 24)
        yy = d['y'][ei]; vv = d['valid'][ei]; last = d['history'][ei, -1]
        # R0 from saved
        with np.load(SRC / f'farm_{farm:02d}' / 'B_speed' / 'evaluation_intervals.npz') as z:
            assert np.array_equal(z['indices'], ei)
            L0 = np.clip(z['lo'], 0, 1); U0 = np.clip(z['hi'], 0, 1)
        for g, v in metrics(yy, L0, U0, vv, last).items():
            rows.append(dict(farm=farm, method='R0', group=g, **v))
        for method in ('R1', 'R2'):
            X, y, S = build_shared_2012(d, method)
            lo, _, _ = fit_predict(X, y, tr_unfolded, 0.05)
            hi, _, _ = fit_predict(X, y, tr_unfolded, 0.95)
            lo, hi = order_quantile_bounds(lo, hi)
            res = simulate(d['origins'], d['ends'], d['y'], lo, hi, d['valid'], es, ei, 'rolling', 84)
            L = np.clip(res['lo'], 0, 1); U = np.clip(res['hi'], 0, 1)
            for g, v in metrics(yy, L, U, vv, last).items():
                rows.append(dict(farm=farm, method=method, group=g, **v))
        print(f'  farm{farm:02d} done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'p2_2012_eval_per_farm.csv', index=False)
    macro = df.groupby(['method', 'group'])[['coverage', 'width', 'IS', 'lower_miss', 'upper_miss']].mean().reset_index()
    macro.to_csv(OUT / 'p2_2012_eval_macro.csv', index=False)
    print('\n===== P2 evaluation 2012 (clipped CQR) =====')
    for g in ('all', 'up_ramp', 'near_zero'):
        line = f'  {g:9s}'
        for m in ('R0', 'R1', 'R2'):
            r = macro[(macro.method == m) & (macro.group == g)].iloc[0]
            line += f"  {m}: cov={r['coverage']*100:5.1f}% w={r['width']*100:5.2f}% IS={r['IS']:.4f}"
        print(line)
    print(f'\n[done] ({time.perf_counter()-began:.1f}s)')


if __name__ == '__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        main()
