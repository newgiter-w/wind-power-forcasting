"""Frozen GEFCom2014 A/B/C experiment; independent of Kelmarsh artifacts.

Run: python gefcom2014_experiment.py --out out_gefcom2014_nwp_v1
CPU only; existing sklearn runtime. Each farm has its own 24 quantile heads.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits
from online_calibration import simulate

ROOT=Path(__file__).resolve().parent
DATA=Path(r'D:\Study\wind_datasets\nwp_benchmarks\gefcom2014_wind\data\Wind')
PKG=dict(max_iter=50,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=0.,random_state=0,early_stopping=False)
METHODS=('A_history','B_nwp100','C_nwp10_100')

def write_json(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load_raw():
    paths=sorted((DATA/'Task 15').rglob('Task15_W_Zone*.csv'))
    paths+=sorted((DATA/'Task 15').rglob('TaskExpVars15_W_Zone*.csv'))
    paths+=[DATA/'Solution to Task 15'/'solution15_W.csv']
    train=pd.concat([pd.read_csv(p) for p in paths if p.name.startswith('Task15_')],ignore_index=True)
    nwp=pd.concat([pd.read_csv(p) for p in paths if p.name.startswith('TaskExpVars')],ignore_index=True)
    answer=pd.read_csv(paths[-1])
    last=nwp.merge(answer,on=['ZONEID','TIMESTAMP'],validate='one_to_one',how='outer',indicator=True)
    assert last['_merge'].eq('both').all()
    full=pd.concat([train,last.drop(columns='_merge')],ignore_index=True)
    assert not full.duplicated(['ZONEID','TIMESTAMP']).any()
    full['timestamp']=pd.to_datetime(full.TIMESTAMP,format='%Y%m%d %H:%M')
    return full,paths

def phase(o):
    if o<pd.Timestamp('2013-01-01'):return 'train'
    if o<pd.Timestamp('2013-04-01'):return 'validation'
    if o<pd.Timestamp('2013-07-01'):return 'calibration'
    return 'test'

def build_farm(raw,zone):
    df=raw[raw.ZONEID.eq(zone)].sort_values('timestamp').set_index('timestamp')
    assert df.index.equals(pd.date_range(df.index.min(),df.index.max(),freq='h'))
    rows=[]; histories=[]; ys=[]; future=[]; base=[]; valid=[]; exclusions=[]
    for o in pd.date_range('2012-01-01','2013-12-31',freq='D'):
        hi=pd.date_range(o-pd.Timedelta(hours=23),o,freq='h')
        ti=pd.date_range(o+pd.Timedelta(hours=1),periods=24,freq='h')
        h=df.TARGETVAR.reindex(hi).to_numpy(float)
        w=df.reindex(ti)[['U10','V10','U100','V100']].to_numpy(float)
        y=df.TARGETVAR.reindex(ti).to_numpy(float)
        reason=[]
        if not np.isfinite(h).all():reason.append('incomplete_observed_history')
        if not np.isfinite(w).all():reason.append('incomplete_nwp')
        if reason:
            exclusions.append(dict(farm=zone,origin=str(o),phase=phase(o),reason=';'.join(reason)))
            continue
        # Only causal input quality determines whether a forecast is issued.
        is_valid=bool(np.isfinite(y).all())
        cal=np.array([np.sin(2*np.pi*o.hour/24),np.cos(2*np.pi*o.hour/24),
                      np.sin(2*np.pi*o.dayofyear/365.25),np.cos(2*np.pi*o.dayofyear/365.25)])
        stat=np.array([h.mean(),h.std(),h[-1],h.min(),h.max(),np.polyfit(np.arange(24),h,1)[0],np.sum(np.abs(np.diff(h))>.1)])
        base.append(np.r_[h,stat,cal]);histories.append(h);future.append(w);ys.append(y);valid.append(is_valid)
        rows.append(dict(farm=zone,origin=str(o),last_observed=str(hi[-1]),
                         target_start=str(ti[0]),target_end=str(ti[-1]),phase=phase(o),
                         target_valid=is_valid,nwp_clock='competition_daily_protocol_inferred',
                         timezone='unspecified_dataset_clock'))
    meta=pd.DataFrame(rows);w=np.array(future);speed10=np.hypot(w[:,:,0],w[:,:,1]);speed100=np.hypot(w[:,:,2],w[:,:,3])
    # Whole future curve is known at daily decision time under competition protocol.
    b100=np.stack([w[:,:,2],w[:,:,3],speed100],axis=2).reshape(len(w),-1)
    both=np.concatenate([w,speed10[:,:,None],speed100[:,:,None]],axis=2).reshape(len(w),-1)
    a=np.array(base)
    X=dict(A_history=a,B_nwp100=np.c_[a,b100],C_nwp10_100=np.c_[a,both])
    y=np.array(ys);v=np.array(valid);hist=np.array(histories)
    origins=pd.to_datetime(meta.origin).to_numpy()
    ends=pd.to_datetime(meta.target_end).to_numpy()
    train=np.flatnonzero(meta.phase.eq('train').to_numpy() & v)
    assert ends[train].max()<=origins[meta.phase.eq('validation')].min()
    assert np.isfinite(y[train]).all()
    for x in X.values():assert np.isfinite(x).all()
    return dict(meta=meta,X=X,y=y,valid=v,history=hist,origins=origins,ends=ends,train=train,exclusions=exclusions)

def masks(y,last):
    delta=np.diff(y,axis=1,prepend=last[:,None])
    return dict(all=np.ones_like(y,dtype=bool),ramp=np.abs(delta)>.1,
                up_ramp=delta>.1,down_ramp=delta<-.1,near_zero=np.abs(y)<=.01)

def metrics(y,lo,hi,valid,last):
    ok=np.broadcast_to(valid[:,None],y.shape)
    inside=(y>=lo)&(y<=hi)
    width=hi-lo
    score=width+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
    out={}
    for group,mask in masks(y,last).items():
        m=mask&ok;n=int(m.sum())
        out[group]=dict(cells=n,origins=int(m.any(axis=1).sum()),
                        coverage=float(inside[m].mean()) if n else None,
                        width=float(width[m].mean()) if n else None,
                        IS=float(score[m].mean()) if n else None,
                        lower_miss=float((y[m]<lo[m]).mean()) if n else None,
                        upper_miss=float((y[m]>hi[m]).mean()) if n else None)
    return out

def preflight(out):
    raw,paths=load_raw()
    farms={z:build_farm(raw,z) for z in range(1,11)}
    sources={str(p):digest(p) for p in paths}
    protocol=dict(dataset='GEFCom2014 Wind Task15 + released December answers',
                  confidence=.9,capacity=1.,history_hours=24,horizon_hours=24,stride_hours=24,
                  model=PKG,quantiles=[.05,.95],seed=0,
                  methods={m:farms[1]['X'][m].shape[1] for m in METHODS},
                  split='origin-date: 2012 train; 2013 Jan-Mar validation; Apr-Jun calibration; Jul-Dec test. Target end may equal next phase first decision time.',
                  selection='No tuning. Validation retained for fixed-model diagnostics only.',
                  cqr='signed symmetric rolling CQR per lead; 84 calendar days; labels only when target_end <= origin; complete-day feedback',
                  interval_support='unclipped for primary metrics; crossing resolved by sorting raw q05/q95; empty CQR interval mapped to raw midpoint as existing simulate()',
                  nwp_limit='daily competition decision time inferred; native ECMWF issue time/latency absent; no claim that physical NWP lead is 1-24h',
                  target_quality='issue if history and NWP finite; score/train complete target days only; never fill target labels',
                  comparison='paired origins and groups across A/B/C; each farm trained/calibrated separately',
                  inference_scope='first frozen dataset backtest; no claim of Kelmarsh improvement or universal gain',
                  python=sys.executable,python_version=platform.python_version(),
                  versions=dict(numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
                  source_sha256=sources,script_sha256=digest(Path(__file__)),
                  calibration_script_sha256=digest(ROOT/'online_calibration.py'))
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=json.loads(protocol_path.read_text(encoding='utf-8'))
        assert old==protocol,'Frozen protocol differs; use a new output directory'
    else:write_json(protocol_path,protocol)
    meta=pd.concat([d['meta'] for d in farms.values()],ignore_index=True)
    meta.to_csv(out/'split_manifest.csv',index=False)
    pd.DataFrame([x for d in farms.values() for x in d['exclusions']]).to_csv(out/'input_exclusions.csv',index=False)
    counts=meta.groupby(['farm','phase']).agg(issued=('origin','size'),valid_targets=('target_valid','sum')).reset_index()
    counts.to_csv(out/'sample_counts.csv',index=False)
    write_json(out/'preflight.json',dict(all_farms=10,paired_feature_origins=True,
              train_label_arrival_assertions='passed',finite_inputs='passed',raw_hashes_recorded=True,
              feature_dimensions=protocol['methods'],sample_counts=counts.to_dict('records')))
    print('PREFLIGHT',protocol['methods'],flush=True)
    return farms

def run(out,farms):
    started=time.perf_counter();records=[];lead_records=[]
    for zone,d in farms.items():
        val_idx=np.flatnonzero(d['meta'].phase.eq('validation').to_numpy())
        test_idx=np.flatnonzero(d['meta'].phase.eq('test').to_numpy())
        seed_idx=np.flatnonzero(d['meta'].phase.eq('calibration').to_numpy())
        assert d['ends'][seed_idx].max()<=d['origins'][test_idx[0]]
        for method in METHODS:
            folder=out/f'farm_{zone:02d}'/method;folder.mkdir(parents=True,exist_ok=True)
            save=folder/'predictions.npz';began=time.perf_counter()
            X=d['X'][method]
            if save.exists():
                with np.load(save) as z:
                    lo=z['raw_lo'];hi=z['raw_hi'];assert np.array_equal(z['origins'],d['origins'])
                    assert np.array_equal(z['y'],d['y'],equal_nan=True)
            else:
                preds=[]
                for q in [.05,.95]:
                    model=MultiOutputRegressor(HistGradientBoostingRegressor(loss='quantile',quantile=q,**PKG))
                    model.fit(X[d['train']],d['y'][d['train']])
                    preds.append(model.predict(X))
                    joblib.dump(model,folder/f'q{q:.2f}.joblib',compress=3)
                lo=np.minimum(*preds);hi=np.maximum(*preds)
                np.savez_compressed(save,raw_lo=lo,raw_hi=hi,y=d['y'],valid=d['valid'],origins=d['origins'],
                                    target_end=d['ends'],last_power=d['history'][:,-1])
            assert np.isfinite(lo).all() and np.isfinite(hi).all()
            res=simulate(d['origins'],d['ends'],d['y'],lo,hi,d['valid'],seed_idx,test_idx,'rolling',84)
            assert (res['max_label_end_used']<=d['origins'][test_idx]).all()
            np.savez_compressed(folder/'calibrated.npz',lo=res['lo'],hi=res['hi'],indices=test_idx,
                                q=res['q_signed'],n_cal=res['n_cal'],max_label_end_used=res['max_label_end_used'])
            pd.DataFrame(res['feedback']).to_csv(folder/'feedback.csv',index=False)
            write_json(folder/'calibration_checks.json',dict(future_labels_used=0,
                       empty_interval_midpoint_count=res['midpoint_expansion_count'],
                       calibration_days_min=int(res['n_cal'].min()),calibration_days_max=int(res['n_cal'].max())))
            for name,idx,L,U in [('validation_raw',val_idx,lo[val_idx],hi[val_idx]),
                                 ('test_raw',test_idx,lo[test_idx],hi[test_idx]),
                                 ('test_cqr',test_idx,res['lo'],res['hi'])]:
                met=metrics(d['y'][idx],L,U,d['valid'][idx],d['history'][idx,-1])
                for group,v in met.items():records.append(dict(farm=zone,method=method,stage=name,group=group,**v))
                if name.startswith('test'):
                    for h in range(24):
                        v=metrics(d['y'][idx,h:h+1],L[:,h:h+1],U[:,h:h+1],d['valid'][idx],d['history'][idx,-1])['all']
                        lead_records.append(dict(farm=zone,method=method,stage=name,lead=h+1,**v))
            pd.DataFrame(records).to_csv(out/'metrics_per_farm.csv',index=False)
            print(f'farm={zone:02d} {method} train={len(d["train"])} test_valid={d["valid"][test_idx].sum()} seconds={time.perf_counter()-began:.1f} elapsed={time.perf_counter()-started:.1f}',flush=True)
    df=pd.DataFrame(records);pd.DataFrame(lead_records).to_csv(out/'metrics_per_lead.csv',index=False)
    macro=df.groupby(['method','stage','group'])[['coverage','width','IS','lower_miss','upper_miss']].mean().reset_index()
    totals=df.groupby(['method','stage','group'])[['cells','origins']].sum().reset_index()
    macro=macro.merge(totals,on=['method','stage','group'])
    macro.to_csv(out/'metrics_macro.csv',index=False)
    # Pooled cells are descriptive; not treated as independent trials.
    pooled=[]
    for (method,stage,group),g in df.groupby(['method','stage','group']):
        row=dict(method=method,stage=stage,group=group,cells=int(g.cells.sum()))
        for col in ['coverage','width','IS','lower_miss','upper_miss']:
            use=g[col].notna()&g.cells.gt(0);row[col]=float(np.average(g.loc[use,col],weights=g.loc[use,'cells']))
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(out/'metrics_pooled.csv',index=False)
    write_json(out/'completion.json',dict(status='complete',farms=10,methods=list(METHODS),
               fits=30,heads=1440,elapsed_seconds=time.perf_counter()-started,
               future_label_assertions='passed',raw_inputs_unchanged=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='out_gefcom2014_nwp_v1');p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    out=Path(a.out);out=out if out.is_absolute() else ROOT/out;out.mkdir(parents=True,exist_ok=True)
    farms=preflight(out)
    if not a.preflight_only:
        with threadpool_limits(limits=2):run(out,farms)

if __name__=='__main__':main()
