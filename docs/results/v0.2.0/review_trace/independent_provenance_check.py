"""Reviewer-only checks, independent of Route A target/calibration functions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
RUN = Path(__file__).resolve().parent.parent
manifest = json.loads((RUN / 'input_manifest.json').read_text(encoding='utf-8'))
raw = [r for r in manifest if r['kind'] == 'raw_data']
hash_checks = []
for item in raw:
    p = Path(item['path'])
    with p.open('rb') as f:
        observed = hashlib.file_digest(f, 'sha256').hexdigest()
    assert observed == item['sha256'], p
    assert p.stat().st_size == item['bytes'], p
    hash_checks.append(str(p))
for item in json.loads((RUN / 'code_manifest.json').read_text(encoding='utf-8')):
    p = ROOT / item['path']
    with p.open('rb') as f:
        observed = hashlib.file_digest(f, 'sha256').hexdigest()
    assert observed == item['sha256'], p
    hash_checks.append(str(p))

p12 = next(Path(r['path']) for r in raw if Path(r['path']).name == 'windpowermeasurements.csv')
d12 = pd.read_csv(p12)
d12.index = pd.to_datetime(d12['date'].astype(str), format='%Y%m%d%H')
tr14 = pd.concat([pd.read_csv(r['path']) for r in raw if Path(r['path']).name.startswith('Task15_W_Zone')], ignore_index=True)
ans14 = pd.read_csv(next(r['path'] for r in raw if Path(r['path']).name == 'solution15_W.csv'))
power14 = pd.concat([tr14[['ZONEID', 'TIMESTAMP', 'TARGETVAR']], ans14[['ZONEID', 'TIMESTAMP', 'TARGETVAR']]], ignore_index=True)
power14['time'] = pd.to_datetime(power14['TIMESTAMP'], format='%Y%m%d %H:%M')
assert not power14.duplicated(['ZONEID', 'time']).any()
summary = []
total_y = total_history = correction_cells = 0
max_err = 0.0
min_rank_margin = 10**9
min_training_gap_hours = float('inf')
for year, nf in [(2014, 10), (2012, 7)]:
    for farm in range(1, nf + 1):
        folder = RUN / f'gefcom{year}' / f'farm_{farm:02d}'
        with np.load(folder / 'predictions.npz') as z:
            p = {k: z[k] for k in z.files}
        meta = pd.read_csv(folder / 'samples.csv')
        phase = meta['phase'].to_numpy()
        times = p['origins'][:, None] + np.arange(1, 25)[None, :] * np.timedelta64(1, 'h')
        history_times = p['origins'][:, None] + np.arange(-23, 1)[None, :] * np.timedelta64(1, 'h')
        source = power14.loc[power14['ZONEID'].eq(farm)].set_index('time')['TARGETVAR'] if year == 2014 else d12[f'wp{farm}']
        y = source.reindex(times.ravel()).to_numpy().reshape(times.shape)
        history = source.reindex(history_times.ravel()).to_numpy().reshape(history_times.shape)
        np.testing.assert_equal(p['y'], y)
        np.testing.assert_equal(p['history'], history)
        np.testing.assert_array_equal(p['ends'], p['origins'] + np.timedelta64(24, 'h'))
        valid = (np.isfinite(y) & (y >= 0) & (y <= 1)).all(axis=1)
        np.testing.assert_equal(p['valid'], valid)
        assert np.isfinite(history).all() and ((history >= 0) & (history <= 1)).all()
        train = np.flatnonzero((phase == 'train') & valid)
        forecast = np.flatnonzero(phase != 'train')
        assert p['ends'][train].max() <= p['origins'][forecast].min()
        gap = float((p['origins'][forecast].min() - p['ends'][train].max()) / np.timedelta64(1, 'h'))
        min_training_gap_hours = min(min_training_gap_hours, gap)
        total_y += y.size
        total_history += history.size
        farm_correction_cells = 0
        for stage, seed_stage in [('validation', 'validation_seed'), ('evaluation', 'validation')]:
            with np.load(folder / f'{stage}_calibration.npz') as z:
                saved = {k: z[k] for k in z.files}
            score_ids = np.flatnonzero(phase == stage)
            seeds = np.flatnonzero((phase == seed_stage) & valid)
            np.testing.assert_array_equal(saved['indices'], score_ids)
            for level, alpha, li, ui in [('0.80', .2, 2, 4), ('0.90', .1, 1, 5), ('0.95', .05, 0, 6)]:
                lower, upper = p['pred'][:, :, li], p['pred'][:, :, ui]
                for method in ['QRF_CQR_static', 'QRF_CQR_rolling84']:
                    prefix = f'{method}_{level}_'
                    for pos, idx in enumerate(score_ids):
                        now = p['origins'][idx]
                        if method == 'QRF_CQR_static':
                            use = seeds
                        else:
                            past = score_ids[:pos]
                            arrived = past[valid[past] & (p['ends'][past] <= now)]
                            use = np.concatenate([seeds, arrived])
                            use = use[p['ends'][use] > now - np.timedelta64(84, 'D')]
                        assert idx not in use and (p['ends'][use] <= now).all()
                        rank = int(np.ceil((len(use) + 1) * (1 - alpha) - 1e-12))
                        assert 1 <= rank <= len(use)
                        min_rank_margin = min(min_rank_margin, len(use) - rank)
                        scores = np.maximum(lower[use] - y[use], y[use] - upper[use])
                        correction = np.sort(scores, axis=0)[rank - 1]
                        l, u = lower[idx] - correction, upper[idx] + correction
                        crossed = l > u
                        mid = (lower[idx] + upper[idx]) / 2
                        l = np.clip(np.where(crossed, mid, l), 0, 1)
                        u = np.clip(np.where(crossed, mid, u), 0, 1)
                        assert saved[prefix + 'n_cal'][pos] == len(use)
                        assert saved[prefix + 'max_label_end'][pos] == p['ends'][use].max()
                        for key, expected in [('lo', l), ('hi', u), ('correction', correction)]:
                            observed = saved[prefix + key][pos]
                            np.testing.assert_array_equal(observed, expected)
                            max_err = max(max_err, float(np.max(np.abs(observed - expected))))
                        farm_correction_cells += correction.size
        correction_cells += farm_correction_cells
        summary.append({'year': year, 'farm': farm, 'y_cells': int(y.size), 'history_cells': int(history.size), 'valid_days': int(valid.sum()), 'correction_cells': int(farm_correction_cells), 'training_gap_hours': gap})
result = {'status': 'pass', 'review_independence': 'same-family', 'acceptance_status': 'provisional', 'independent_functions': True, 'raw_hashes_checked': len(raw), 'code_hashes_checked': len(hash_checks) - len(raw), 'farms_checked': len(summary), 'target_cells_checked': total_y, 'history_cells_checked': total_history, 'calibration_correction_cells_checked': correction_cells, 'max_absolute_calibration_error': max_err, 'minimum_n_minus_rank': min_rank_margin, 'minimum_training_gap_hours': min_training_gap_hours, 'details': summary}
target = RUN / 'review_trace' / 'independent_provenance_results.json'
target.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k != 'details'}, indent=2))
