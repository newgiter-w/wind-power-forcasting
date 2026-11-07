"""P0: frozen-QRF replay, two fixed refits, independent causal CQR, diagnostics.

Run only with tools/run_route_a.py. Legacy modules imported here only supply
read-only data construction / reference calibration; their main() is never run.
"""
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '2'
from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import time
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
import quantile_forest
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from gefcom_protocol_v2 import prepare, index
from run_gefcom_tail_dynamic import base_matrix
from online_calibration import simulate
from route_a.paths import PROJECT_ROOT as ROOT, new_run_dir, require_local_write

Q = np.array([.025, .05, .1, .5, .9, .95, .975])
ALPHAS = (.2, .1, .05)
METHODS = ('QRF_raw', 'QRF_CQR_static', 'QRF_CQR_rolling84')
OLD = ROOT / 'out_gefcom_revision_20261104/qrf'
INPUT = ROOT / 'out_gefcom_pooling_20261105/inputs'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(p, value):
    require_local_write(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def predict(model, x):
    chunks = []
    for start in range(0, len(x), 512):
        chunks.append(model.predict(x[start:start+512], quantiles=Q.tolist(),
            weighted_quantile=True, weighted_leaves=True, aggregate_leaves_first=True))
    return np.concatenate(chunks)


def causal_cqr(d, pred, seed_ids, ids, alpha, rolling):
    """Independent event-time implementation, not calling legacy simulate()."""
    qi = int(np.argmin(abs(Q-alpha/2)))
    qj = int(np.argmin(abs(Q-(1-alpha/2))))
    raw_l, raw_u = pred[..., qi], pred[..., qj]
    lo, hi, corrections, counts, max_end = [], [], [], [], []
    consumed = list(seed_ids[d['valid'][seed_ids]])
    pending = []
    cross_count = 0
    for idx in ids:
        now = d['origins'][idx]
        if rolling:
            for prior in pending:
                if d['ends'][prior] <= now and d['valid'][prior]:
                    consumed.append(prior)
            pending = [prior for prior in pending if d['ends'][prior] > now]
        used = np.asarray(consumed, dtype=int)
        assert len(used) and (d['ends'][used] <= now).all() and idx not in used
        if rolling:
            used = used[d['ends'][used] > now-np.timedelta64(84, 'D')]
        scores = np.maximum(raw_l[used]-d['y'][used], d['y'][used]-raw_u[used])
        assert np.isfinite(scores).all()
        rank = int(np.ceil((len(used)+1)*(1-alpha)-1e-12))
        if not 1 <= rank <= len(used):
            raise ValueError('Insufficient calibration data for finite endpoint')
        correction = np.sort(scores, axis=0)[rank-1]
        l, u = raw_l[idx]-correction, raw_u[idx]+correction
        crossed = l > u
        mid = (raw_l[idx]+raw_u[idx])/2
        cross_count += int(crossed.sum())
        lo.append(np.clip(np.where(crossed, mid, l), 0, 1))
        hi.append(np.clip(np.where(crossed, mid, u), 0, 1))
        corrections.append(correction)
        counts.append(len(used))
        max_end.append(d['ends'][used].max())
        pending.append(idx)
    return dict(lo=np.array(lo), hi=np.array(hi), correction=np.array(corrections),
        n_cal=np.array(counts), max_label_end=np.array(max_end), midpoint_collapses=cross_count)


def metric_arrays(y, lo, hi, alpha):
    return dict(coverage=((y >= lo) & (y <= hi)).astype(float), width=hi-lo,
        IS=hi-lo+2/alpha*(np.maximum(lo-y, 0)+np.maximum(y-hi, 0)),
        lower_miss=(y < lo).astype(float), upper_miss=(y > hi).astype(float))


def groups(d, ids):
    y = d['y'][ids]
    delta = np.diff(y, axis=1, prepend=d['history'][ids, -1, None])
    result = [('regime', 'all', np.ones(y.shape, bool)),
        ('regime', 'up_ramp', delta > .1), ('regime', 'down_ramp', delta < -.1),
        ('regime', 'near_zero', y <= .01)]
    for lo, hi, label in [(0, .01, '0_to_.01'), (.01, .2, '.01_to_.2'),
                          (.2, .8, '.2_to_.8'), (.8, 1, '.8_to_1')]:
        result.append(('power', label, (y >= lo if lo == 0 else y > lo) & (y <= hi)))
    for h in range(24):
        result.append(('lead', str(h+1), np.broadcast_to(np.arange(24) == h, y.shape)))
    dates = pd.DatetimeIndex(d['origins'][ids])
    # Group by target valid time, so h24 belongs to next calendar day/month.
    target_times = d['origins'][ids, None] + np.arange(1, 25)[None, :]*np.timedelta64(1, 'h')
    month = target_times.astype('datetime64[M]').astype(str)
    for m in np.unique(month):
        result.append(('month', str(m), month == m))
    mn = pd.DatetimeIndex(target_times.ravel()).month.to_numpy().reshape(y.shape)
    for label, months in [('DJF', [12,1,2]), ('MAM', [3,4,5]), ('JJA', [6,7,8]), ('SON', [9,10,11])]:
        result.append(('season', label, np.isin(mn, months)))
    return result


def score_rows(rows, daily, year, farm, stage, d, ids, pred, methods):
    y = d['y'][ids]
    valid = d['valid'][ids, None] & np.isfinite(y)
    median = pred[ids, :, 3]
    delta = np.diff(y, axis=1, prepend=d['history'][ids, -1, None])
    for name, endpoints in methods.items():
        wis = .5*np.abs(y-median)
        for ai, alpha in enumerate(ALPHAS):
            lo, hi = endpoints[ai]
            a = metric_arrays(y, lo, hi, alpha)
            wis = wis+alpha/2*a['IS']
            daily[year, farm, stage, name, alpha] = dict(origins=d['origins'][ids],
                n=valid.sum(axis=1), **{k:np.where(valid,v,0).sum(axis=1) for k,v in a.items()})
            for dimension, group, mask in groups(d, ids):
                m = mask & valid
                if not m.any():
                    continue
                rows.append(dict(year=year, farm=farm, stage=stage, method=name,
                    confidence=1-alpha, dimension=dimension, group=group, cells=int(m.sum()),
                    days=int(m.any(axis=1).sum()), **{k:float(v[m].mean()) for k,v in a.items()}))
        wis = wis / 3.5
        good = d['valid'][ids]
        summary = dict(year=year, farm=farm, stage=stage, method=name,
            cells=int(valid.sum()), WIS=float(wis[valid].mean()),
            median_MAE=float(np.abs(y-median)[valid].mean()),
            persistence_MAE=float(np.abs(y-d['history'][ids,-1,None])[valid].mean()),
            up_ramp_cells=int(((delta>.1)&valid).sum()), down_ramp_cells=int(((delta<-.1)&valid).sum()))
        for ai, alpha in enumerate(ALPHAS):
            lo, hi = endpoints[ai]
            summary[f'day_coverage_{1-alpha:.2f}'] = float(((y>=lo)&(y<=hi)).all(axis=1)[good].mean())
        summary['nested_80_in_90_violations'] = int((((endpoints[0][0] < endpoints[1][0]) | (endpoints[0][1] > endpoints[1][1])) & valid).sum())
        summary['nested_90_in_95_violations'] = int((((endpoints[1][0] < endpoints[2][0]) | (endpoints[1][1] > endpoints[2][1])) & valid).sum())
        yield summary


def bootstrap(daily, out):
    rows = []
    rng = np.random.default_rng(20261107)
    for year in (2014,2012):
        farms = sorted({k[1] for k in daily if k[0]==year})
        for alpha in ALPHAS:
            dates = np.unique(np.concatenate([daily[year,f,'evaluation',METHODS[0],alpha]['origins'] for f in farms]))
            n = len(dates)
            draws = ((rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1))+np.arange(14)) % n).reshape(1000,-1)[:,:n]
            for other in METHODS[1:]:
                for metric in ('coverage','width','IS'):
                    values = {}
                    for method in (METHODS[0],other):
                        estimates=[];observed=[]
                        for f in farms:
                            d = daily[year,f,'evaluation',method,alpha]
                            pos=np.searchsorted(dates,d['origins'])
                            sums=np.zeros(n);counts=np.zeros(n)
                            sums[pos]=d[metric];counts[pos]=d['n']
                            denom=counts[draws].sum(axis=1)
                            assert (denom>0).all()
                            estimates.append(sums[draws].sum(axis=1)/denom)
                            observed.append(sums.sum()/counts.sum())
                        values[method]=(np.mean(estimates,axis=0),float(np.mean(observed)))
                    diff=values[other][0]-values[METHODS[0]][0]
                    ci=np.quantile(diff,[.025,.975])
                    rows.append(dict(year=year,confidence=1-alpha,method=other,reference=METHODS[0],
                        metric=metric,delta=values[other][1]-values[METHODS[0]][1],
                        ci_low=float(ci[0]),ci_high=float(ci[1]),block_days=14,draws=1000))
    pd.DataFrame(rows).to_csv(out/'paired_blocks.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out');args=parser.parse_args()
    out=require_local_write(Path(args.out).resolve()) if args.out else new_run_dir('p0_baseline')
    out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((ROOT/'route_a/p0_protocol.json').read_text(encoding='utf-8'))
    assert not (out/'completion.json').exists(), 'Never overwrite a completed run'
    write(out/'protocol.json',protocol)
    print('OUTPUT',out,flush=True)
    started=time.perf_counter();rows=[];summaries=[];checks=[];quality=[];pins=[];daily={};provenance=[];support=[]
    write(out/'environment.json',dict(python=platform.python_version(),executable=os.sys.executable,
        numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,quantile_forest=quantile_forest.__version__,
        qrf_module=quantile_forest.__file__,guard=os.environ.get('ROUTE_A_GUARD'),cpu=platform.processor()))
    for year,nfarms in ((2014,10),(2012,7)):
        farms,sources=prepare(year)
        assert len(farms)==nfarms
        for p in sources:
            provenance.append(dict(kind='raw_data',path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size))
        for farm,d in sorted(farms.items()):
            t0=time.perf_counter();folder=out/f'gefcom{year}/farm_{farm:02d}';folder.mkdir(parents=True,exist_ok=True)
            old=OLD/f'gefcom{year}/farm_{farm:02d}/leaf_10/seed_0'
            cached=INPUT/f'gefcom{year}/farm_{farm:02d}'
            with np.load(cached/'samples.npz') as z:
                for k in ('y','history','valid','origins','ends'):
                    np.testing.assert_equal(d[k],z[k])
            meta=pd.read_csv(cached/'meta.csv')
            np.testing.assert_array_equal(meta.phase,d['meta'].phase)
            x=base_matrix(d,year);tr=index(d,'train',True)
            tm=np.repeat(np.isin(np.arange(len(d['y'])),tr),24)
            forecast=np.flatnonzero(~d['meta'].phase.eq('train').to_numpy())
            assert d['ends'][tr].max()<=d['origins'][forecast].min()
            fitted=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(x[tm])
            pack=joblib.load(old/'model.joblib');model=pack['model'];model.n_jobs=2
            np.testing.assert_equal(fitted.statistics_,pack['imputer'].statistics_)
            np.testing.assert_array_equal(fitted.indicator_.features_,pack['imputer'].indicator_.features_)
            dense=pack['imputer'].transform(x)
            with np.load(cached/'features.npz') as z:
                np.testing.assert_equal(x,z['GBM'])
                np.testing.assert_equal(dense.astype('float32'),z['QRF'])
            fit=json.loads((old/'fit.json').read_text())
            assert fit['train_cells']==tm.sum() and fit['features']==dense.shape[1]
            assert model.n_estimators==500 and model.min_samples_leaf==10 and model.random_state==0 and model.max_samples_leaf is None
            pred=np.full((len(d['y']),24,len(Q)),np.nan)
            flat=(forecast[:,None]*24+np.arange(24)).ravel()
            pred[forecast]=predict(model,dense[flat]).reshape(len(forecast),24,len(Q))
            assert np.isfinite(pred[forecast]).all() and (np.diff(pred[forecast],axis=-1)>=-1e-12).all()
            with np.load(old/'heads.npz') as z:
                np.testing.assert_equal(z['origins'],d['origins']);np.testing.assert_array_equal(z['forecast_indices'],forecast)
                raw_error=max(np.max(abs(pred[forecast,:,1]-z['lo'][forecast])),np.max(abs(pred[forecast,:,5]-z['hi'][forecast])))
                np.testing.assert_allclose(pred[forecast,:,1],z['lo'][forecast],rtol=0,atol=1e-12)
                np.testing.assert_allclose(pred[forecast,:,5],z['hi'][forecast],rtol=0,atol=1e-12)
            witness=dict(year=year,farm=farm,raw_max_abs_error=float(raw_error),train_cells=int(tm.sum()),features=dense.shape[1],raw_data_equal=True,train_only_imputer_equal=True)
            if farm==1:
                print('REFIT_START',year,farm,flush=True)
                refit=RandomForestQuantileRegressor(n_estimators=500,min_samples_leaf=10,
                    max_features=1.,max_samples_leaf=None,random_state=0,n_jobs=2).fit(dense[tm],d['y'].ravel()[tm])
                fresh=predict(refit,dense[flat]).reshape(len(forecast),24,len(Q))
                diff=float(np.max(abs(fresh-pred[forecast])))
                np.testing.assert_allclose(fresh,pred[forecast],rtol=0,atol=1e-12)
                witness['refit_max_abs_error']=diff
                witness['refit_tree_structure_equal']=all(np.array_equal(a.tree_.threshold,b.tree_.threshold) and np.array_equal(a.tree_.feature,b.tree_.feature) for a,b in zip(model.estimators_,refit.estimators_))
                assert witness['refit_tree_structure_equal']
                joblib.dump({'model':refit,'imputer':fitted},folder/'refit.joblib',compress=3)
                del refit,fresh
            np.savez_compressed(folder/'predictions.npz',quantiles=Q,pred=pred,origins=d['origins'],y=d['y'],history=d['history'],valid=d['valid'],ends=d['ends'])
            d['meta'].to_csv(folder/'samples.csv',index=False)
            d['calendar'].to_csv(folder/'calendar.csv',index=False)
            for phase,g in d['calendar'].groupby('phase'):
                issued=d['meta'].phase.eq(phase).to_numpy()
                quality.append(dict(year=year,farm=farm,phase=phase,calendar_days=len(g),issued_days=int(issued.sum()),
                    valid_label_days=int(d['valid'][issued].sum()),input_service_rate=float(issued.sum()/len(g)),
                    bad_history_days=int((~g.history_valid).sum()),bad_nwp_days=int((~g.nwp_valid).sum()),
                    train_label_cells=int(len(tr)*24) if phase=='train' else 0))
            for stage in ('validation','evaluation'):
                ids=index(d,stage);seed=index(d,'validation_seed' if stage=='validation' else 'validation')
                allends={name:[] for name in METHODS};cal_record={}
                for alpha in ALPHAS:
                    qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
                    allends[METHODS[0]].append((pred[ids,:,qi],pred[ids,:,qj]))
                    for name,rolling in zip(METHODS[1:],(False,True)):
                        result=causal_cqr(d,pred,seed,ids,alpha,rolling)
                        allends[name].append((result['lo'],result['hi']))
                        key=f'{name}_{1-alpha:.2f}'
                        for k in ('lo','hi','correction','n_cal','max_label_end'):
                            cal_record[f'{key}_{k}']=result[k]
                        witness[f'{stage}_{key}_min_n']=int(result['n_cal'].min())
                        witness[f'{stage}_{key}_midpoint_collapses']=result['midpoint_collapses']
                        if alpha==.1 and rolling:
                            ref=simulate(d['origins'],d['ends'],d['y'],pred[...,1],pred[...,5],d['valid'],seed,ids,'rolling',84)
                            with np.load(old/f'{stage}.npz') as z:
                                np.testing.assert_array_equal(ids,z['indices'])
                                for endpoint in ('lo','hi'):
                                    expected=np.clip(z[endpoint],0,1)
                                    np.testing.assert_allclose(result[endpoint],expected,rtol=0,atol=1e-12)
                                    np.testing.assert_allclose(result[endpoint],np.clip(ref[endpoint],0,1),rtol=0,atol=1e-12)
                                err=max(np.max(abs(result['lo']-np.clip(z['lo'],0,1))),np.max(abs(result['hi']-np.clip(z['hi'],0,1))))
                                witness[f'{stage}_cqr_max_abs_error']=float(err)
                np.savez_compressed(folder/f'{stage}_calibration.npz',indices=ids,**cal_record)
                summaries.extend(score_rows(rows,daily,year,farm,stage,d,ids,pred,allends))
                yy=d['y'][ids];valid=np.broadcast_to(d['valid'][ids,None],yy.shape)
                for qi,q in enumerate(Q):
                    err=yy-pred[ids,:,qi];loss=np.maximum(q*err,(q-1)*err)
                    pins.append(dict(year=year,farm=farm,stage=stage,quantile=float(q),pinball=float(loss[valid].mean())))
                ws=np.hypot(d['nwp'][:,:,-2],d['nwp'][:,:,-1]);ws_train=ws[tr]
                low,high=np.quantile(ws_train,[.01,.99])
                outside=(ws[ids]<low)|(ws[ids]>high)
                support.append(dict(year=year,farm=farm,stage=stage,wind_train_p01=float(low),wind_train_p99=float(high),
                    wind_outside_train_p01_p99_fraction=float(outside[valid].mean()),
                    target_zero_fraction=float((yy[valid]==0).mean()),target_near_zero_fraction=float((yy[valid]<=.01).mean()),
                    target_std=float(yy[valid].std()),train_target_std=float(d['y'][tr].std()),
                    nwp_age_max_hours=float(d['meta'].iloc[ids].nwp_age_hours.max()),
                    raw_nan_features=int(np.isnan(x.reshape(len(d['y']),24,-1)[ids]).sum())))
            for p in (old/'model.joblib',old/'heads.npz',old/'fit.json',old/'validation.npz',old/'evaluation.npz',cached/'samples.npz',cached/'features.npz'):
                provenance.append(dict(kind='copied_baseline',path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
            checks.append(witness);write(folder/'reproduction.json',witness)
            print('FARM_DONE',year,farm,'seconds',round(time.perf_counter()-t0,2),'raw_error',raw_error,flush=True)
            del model,pack,dense,x,pred;gc.collect()
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_per_farm.csv',index=False)
    keys=['year','stage','method','confidence','dimension','group']
    frame.groupby(keys)[['coverage','width','IS','lower_miss','upper_miss']].mean().reset_index().to_csv(out/'metrics_macro.csv',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(out/'scores_per_farm.csv',index=False)
    summary.groupby(['year','stage','method']).mean(numeric_only=True).drop(columns=['farm']).reset_index().to_csv(out/'scores_macro.csv',index=False)
    pd.DataFrame(quality).to_csv(out/'data_quality.csv',index=False)
    pd.DataFrame(pins).to_csv(out/'pinball_per_farm.csv',index=False)
    pd.DataFrame(support).to_csv(out/'data_support.csv',index=False)
    write(out/'reproduction.json',checks);write(out/'input_manifest.json',provenance)
    bootstrap(daily,out)
    source_files=['route_a/p0_baseline.py','route_a/p0_protocol.json','tools/run_route_a.py',
        'tools/runtime_guard/sitecustomize.py','gefcom_protocol_v2.py','gefcom2014_experiment.py',
        'run_gefcom_tail_dynamic.py','strong_baselines.py','run_p2_2012.py','online_calibration.py']
    write(out/'code_manifest.json',[dict(path=p,sha256=sha(ROOT/p)) for p in source_files])
    write(out/'completion.json',dict(status='complete',farms=len(checks),frozen_replays=17,from_scratch_refits=2,
        new_method_training=0,elapsed_seconds=time.perf_counter()-started,scope='retrospective, seed0',version='0.2.0'))
    print('COMPLETE',out,'seconds',round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
