"""Event-ordered calibration of a frozen forecaster, with complete-day feedback."""
from __future__ import annotations

import numpy as np
import pandas as pd


def alpha_update(alpha, miss, gamma, projected):
    updated = np.asarray(alpha) + gamma * (.1 - np.asarray(miss))
    if projected:
        clipped = np.clip(updated, .02, .5)
        return clipped, int(np.count_nonzero(updated != clipped))
    return updated, 0


def score_quantile(scores, alpha, finite_support=False):
    """Per-lead finite-sample order statistic, including extended-set endpoints."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Finite, nonempty (days, leads) calibration matrix required')
    n, h = scores.shape
    requested = np.broadcast_to(np.asarray(alpha, dtype=float), (h,)).copy()
    effective = np.maximum(requested, 1/(n+1)) if finite_support else requested.copy()
    ranks = np.ceil((n+1)*(1-effective)-1e-12).astype(int)
    q = np.empty(h)
    sorted_scores = np.sort(scores, axis=0)
    for lead in range(h):
        if effective[lead] <= 0 or ranks[lead] > n:
            q[lead] = np.inf
        elif effective[lead] >= 1 or ranks[lead] < 1:
            q[lead] = -np.inf
        else:
            q[lead] = sorted_scores[ranks[lead]-1, lead]
    return q, effective, int(np.count_nonzero(effective != requested))


def simulate_independent(origins, target_end, y, raw_lo, raw_hi, target_valid,
                         seed_indices, score_indices, method='rolling', window_days=84):
    """C1: independent lower/upper tail calibration (fixed 5%/5% budget).

    Lower score sL = raw_lo - y (miss below), upper score sU = y - raw_hi (miss
    above). Each tail calibrated at the 95% order statistic independently:
        qL = Q95({sL}), qU = Q95({sU});  interval = [raw_lo - qL, raw_hi + qU].
    This replaces the symmetric single correction q and lets the upper bound
    widen for upward ramps while the lower bound drops for near-zero power.
    """
    origins = np.asarray(origins, dtype='datetime64[ns]')
    ends = np.asarray(target_end, dtype='datetime64[ns]')
    seed_indices, score_indices = np.asarray(seed_indices), np.asarray(score_indices)
    if not len(score_indices) or method not in ('global', 'rolling'):
        raise ValueError('Unknown method or empty scoring period')
    first = origins[score_indices[0]]
    ids_l, vals_l, ids_u, vals_u = [], [], [], []
    for idx in seed_indices:
        if ends[idx] > first:
            raise ValueError('Seed label not yet available at first forecast')
        if target_valid[idx]:
            if not np.isfinite(y[idx]).all():
                raise ValueError('Valid seed has missing target')
            ids_l.append(int(idx)); vals_l.append(raw_lo[idx] - y[idx])
            ids_u.append(int(idx)); vals_u.append(y[idx] - raw_hi[idx])
    if len(ids_l) < 9:
        raise ValueError('Too few valid calibration days for finite nominal 95% tails')
    seed_ids = np.asarray(ids_l)
    global_qL, _, _ = score_quantile(np.stack(vals_l), .05)
    global_qU, _, _ = score_quantile(np.stack(vals_u), .05)
    n, h = len(score_indices), y.shape[1]
    output = {k: np.empty((n, h)) for k in ('lo', 'hi', 'q_lower', 'q_upper')}
    n_cal = np.empty(n, dtype=int)
    max_used_end = np.empty(n, dtype='datetime64[ns]')
    issued, consumed = [], 0
    feedback = []
    midpoint_expansions = 0
    for position, idx in enumerate(score_indices):
        now = origins[idx]
        while consumed < len(issued) and ends[issued[consumed][0]] <= now:
            past, past_position = issued[consumed]
            consumed += 1
            valid_feedback = bool(target_valid[past])
            if valid_feedback:
                if not np.isfinite(y[past]).all():
                    raise ValueError('Invalid observed target at feedback event')
                ids_l.append(int(past)); vals_l.append(raw_lo[past] - y[past])
                ids_u.append(int(past)); vals_u.append(y[past] - raw_hi[past])
            feedback.append({'source_index': int(past), 'source_origin': str(origins[past]),
                             'source_target_end': str(ends[past]), 'processed_at': str(now),
                             'valid_feedback': valid_feedback,
                             'action': 'observe_frozen' if method == 'global' else 'update_if_valid'})
        if method == 'global':
            used = seed_ids
            qL, qU = global_qL.copy(), global_qU.copy()
        else:
            ids = np.asarray(ids_l)
            keep = (ends[ids] > now - np.timedelta64(int(window_days), 'D')) & (ends[ids] <= now)
            used = ids[keep]
            if len(used) < 9:
                raise ValueError(f'Only {len(used)} arrived scores in {window_days}-day window at {now}')
            qL, _, _ = score_quantile(np.stack(vals_l)[keep], .05)
            qU, _, _ = score_quantile(np.stack(vals_u)[keep], .05)
        if np.any(ends[used] > now) or int(idx) in used:
            raise AssertionError('Future or self label entered calibration')
        n_cal[position] = len(used)
        max_used_end[position] = ends[used].max()
        lower, upper = raw_lo[idx] - qL, raw_hi[idx] + qU
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError('Main operational method has nonfinite interval')
        empty = lower > upper
        midpoint_expansions += int(empty.sum())
        center = (raw_lo[idx] + raw_hi[idx]) / 2
        lower = np.where(empty, center, lower)
        upper = np.where(empty, center, upper)
        output['lo'][position] = lower
        output['hi'][position] = upper
        output['q_lower'][position] = qL
        output['q_upper'][position] = qU
        issued.append((int(idx), position))
    output.update(indices=score_indices.copy(), n_cal=n_cal, max_label_end_used=max_used_end,
                  feedback=feedback, method=method, window_days=window_days,
                  midpoint_expansion_count=midpoint_expansions,
                  state_projection_count=0, support_projection_count=0)
    return output


def simulate(origins, target_end, y, raw_lo, raw_hi, target_valid, seed_indices,
             score_indices, method, window_days=84, gamma=0.):
    """Always forecast first; inspect a scored day's label/quality only on arrival.

    Dates and indices refer to the common raw prediction cache. Initial seed
    labels are checked against the first issue time. Other feedback enters
    only after target_end <= current issue time. All 24 leads update together.
    """
    origins = np.asarray(origins, dtype='datetime64[ns]')
    ends = np.asarray(target_end, dtype='datetime64[ns]')
    seed_indices, score_indices = np.asarray(seed_indices), np.asarray(score_indices)
    if not len(score_indices) or method not in ('global', 'rolling', 'aci_projected', 'aci_unprojected'):
        raise ValueError('Unknown method or empty scoring period')
    first = origins[score_indices[0]]
    history_ids, history_values = [], []
    for idx in seed_indices:
        if ends[idx] > first:
            raise ValueError('Seed label not yet available at first forecast')
        if target_valid[idx]:
            if not np.isfinite(y[idx]).all():
                raise ValueError('Valid seed has missing target')
            history_ids.append(int(idx))
            history_values.append(np.maximum(raw_lo[idx]-y[idx], y[idx]-raw_hi[idx]))
    if len(history_ids) < 9:
        raise ValueError('Too few valid calibration days for finite nominal 90% correction')
    seed_ids = np.asarray(history_ids)
    global_q, _, _ = score_quantile(np.stack(history_values), .1)
    n, h = len(score_indices), y.shape[1]
    output = {k:np.empty((n,h)) for k in ('lo','hi','q_signed','q_used','alpha_requested','alpha_used')}
    n_cal = np.empty(n, dtype=int)
    max_used_end = np.empty(n, dtype='datetime64[ns]')
    alpha = np.full(h,.1)
    issued, consumed = [], 0
    feedback = []
    state_projection = support_projection = midpoint_expansions = 0
    is_aci = method.startswith('aci_')
    projected = method == 'aci_projected'
    global_indices = []
    for position, idx in enumerate(score_indices):
        now = origins[idx]
        while consumed < len(issued) and ends[issued[consumed][0]] <= now:
            past, past_position = issued[consumed]
            consumed += 1
            # This is the first and only quality/label inspection for this issued day.
            valid_feedback = bool(target_valid[past])
            if valid_feedback:
                if not np.isfinite(y[past]).all():
                    raise ValueError('Invalid observed target at feedback event')
                value = np.maximum(raw_lo[past]-y[past], y[past]-raw_hi[past])
                history_ids.append(int(past)); history_values.append(value)
                if is_aci:
                    miss = ~((y[past]>=output['lo'][past_position]) &
                             (y[past]<=output['hi'][past_position]))
                    alpha, changed = alpha_update(alpha,miss.astype(float),gamma,projected)
                    state_projection += changed
                else:
                    miss = None
            else:
                miss = None
            feedback.append({'source_index':int(past),'source_origin':str(origins[past]),
                             'source_target_end':str(ends[past]),'processed_at':str(now),
                             'valid_feedback':valid_feedback,
                             'miss_count':None if miss is None else int(miss.sum()),
                             'action':'observe_frozen' if method=='global' else 'update_if_valid'})
        if method == 'global':
            used = seed_ids
            q = global_q.copy(); requested = np.full(h,.1); effective=requested.copy()
        else:
            ids = np.asarray(history_ids)
            keep = (ends[ids] > now-np.timedelta64(int(window_days),'D')) & (ends[ids]<=now)
            used = ids[keep]
            if len(used)<9:
                raise ValueError(f'Only {len(used)} arrived scores in {window_days}-day window at {now}')
            values = np.stack(history_values)[keep]
            requested = alpha.copy() if is_aci else np.full(h,.1)
            q,effective,changed = score_quantile(values,requested,finite_support=projected)
            support_projection += changed
        if np.any(ends[used]>now) or int(idx) in used:
            raise AssertionError('Future or self label entered calibration')
        n_cal[position]=len(used);max_used_end[position]=ends[used].max()
        lower,upper=raw_lo[idx]-q,raw_hi[idx]+q
        q_used=q.copy()
        if method!='aci_unprojected':
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise ValueError('Main operational method has nonfinite interval')
            empty=lower>upper
            midpoint_expansions += int(empty.sum())
            center=(raw_lo[idx]+raw_hi[idx])/2
            lower=np.where(empty,center,lower);upper=np.where(empty,center,upper)
            q_used=np.where(empty,-(raw_hi[idx]-raw_lo[idx])/2,q_used)
        for key,val in (('lo',lower),('hi',upper),('q_signed',q),('q_used',q_used),
                        ('alpha_requested',requested),('alpha_used',effective)):
            output[key][position]=val
        issued.append((int(idx),position));global_indices.append(int(idx))
    output.update(indices=np.asarray(global_indices),n_cal=n_cal,max_label_end_used=max_used_end,
                  feedback=feedback,method=method,window_days=window_days,gamma=gamma,
                  state_projection_count=state_projection,support_projection_count=support_projection,
                  midpoint_expansion_count=midpoint_expansions)
    return output
