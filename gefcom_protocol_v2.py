"""Causal daily GEFCom experiments, representation selection, and frozen transfer.

python -u gefcom_protocol_v2.py --dataset 2014
python -u gefcom_protocol_v2.py --dataset 2012  (requires frozen 2014 policy)
"""
from __future__ import annotations
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
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
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from online_calibration import simulate
from gefcom2014_experiment import load_raw,metrics,masks

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'out_gefcom_protocol_v2'
DATA=Path(r'D:\Study\wind_datasets\nwp_benchmarks')
PKG=dict(max_iter=50,max_leaf_nodes=31,min_samples_leaf=20,l2_regularization=0.,random_state=0,early_stopping=False)
METHODS14=['A_history','B_uv','B_speed','C_uv','C_speed']
METHODS12=['A_history','B_uv','B_speed']
GROUPS=['all','ramp','up_ramp','down_ramp','near_zero']

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def clock_phase(origin,year):
    o=pd.Timestamp(origin)
    if year==2014:
        if o<pd.Timestamp('2013-01-01'):return 'train'
        if o<pd.Timestamp('2013-04-01'):return 'validation_seed'
        if o<pd.Timestamp('2013-07-01'):return 'validation'
        return 'evaluation'
    if o<pd.Timestamp('2011-01-01'):return 'train'
    if o<pd.Timestamp('2011-04-01'):return 'validation_seed'
    if o<pd.Timestamp('2011-07-01'):return 'validation'
    return 'evaluation'

def base_features(h,o):
    x=np.arange(24);o=pd.Timestamp(o)
    stats=[h.mean(),h.std(),h[-1],h.min(),h.max(),np.polyfit(x,h,1)[0],np.sum(np.abs(np.diff(h))>.1)]
    cal=[np.sin(2*np.pi*o.dayofyear/365.25),np.cos(2*np.pi*o.dayofyear/365.25),
         np.sin(2*np.pi*o.hour/24),np.cos(2*np.pi*o.hour/24)]
    return np.r_[h,stats,cal]

def features(d,method,year):
    a=np.array([base_features(h,o) for h,o in zip(d['history'],d['origins'])])
    if method=='A_history':return a
    w=d['nwp'];u=w[:,:,-2];v=w[:,:,-1]
    fields=[u,v]
    if method.startswith('C_'):fields=[w[:,:,0],w[:,:,1],u,v]
    if method.endswith('speed'):
        if method.startswith('C_'):fields.append(np.hypot(w[:,:,0],w[:,:,1]))
        fields.append(np.hypot(u,v))
    extra=[f for f in fields]
    if year==2012:extra.append(d['meta'].nwp_age_hours.to_numpy()[:,None]/48.)
    return np.c_[a,*extra]

def assemble(power,weather,origins,year,farm,source_fields,age=None,issue=None):
    rows=[];hist=[];targets=[];nwps=[]
    for o in origins:
        hi=pd.date_range(o-pd.Timedelta(hours=23),o,freq='h')
        ti=pd.date_range(o+pd.Timedelta(hours=1),periods=24,freq='h')
        h=power.reindex(hi).to_numpy(float);y=power.reindex(ti).to_numpy(float)
        if year==2014:w=weather.reindex(ti)[source_fields].to_numpy(float)
        else:w=weather.get(o,np.full((24,2),np.nan))
        hgood=bool(np.isfinite(h).all() and ((h>=0)&(h<=1)).all())
        wgood=bool(np.isfinite(w).all());ygood=bool(np.isfinite(y).all() and ((y>=0)&(y<=1)).all())
        rows.append(dict(farm=farm,origin=str(o),target_start=str(ti[0]),target_end=str(ti[-1]),
                         phase=clock_phase(o,year),history_valid=hgood,nwp_valid=wgood,target_valid=ygood,
                         input_valid=hgood and wgood,nwp_age_hours=float(age[o]) if age and o in age else 0.,
                         nwp_issue=str(issue[o]) if issue and o in issue else '',
                         clock='dataset clock, timezone unspecified',
                         exclusion='' if hgood and wgood else ('history_missing' if not hgood else 'nwp_unavailable')))
        hist.append(h);targets.append(y);nwps.append(w)
    meta=pd.DataFrame(rows);keep=meta.input_valid.to_numpy()
    # Input eligibility is decided without target quality; preserve all calendar exclusions.
    out=dict(meta=meta[keep].reset_index(drop=True),calendar=meta,history=np.array(hist)[keep],
             y=np.array(targets)[keep],nwp=np.array(nwps)[keep])
    out['origins']=pd.to_datetime(out['meta'].origin).to_numpy()
    out['ends']=pd.to_datetime(out['meta'].target_end).to_numpy()
    out['valid']=out['meta'].target_valid.to_numpy(bool)
    return out

def prepare(year):
    farms={};sources=[]
    if year==2014:
        raw,sources=load_raw()
        for farm,g in raw.groupby('ZONEID'):
            g=g.sort_values('timestamp').set_index('timestamp')
            assert g.index.equals(pd.date_range(g.index.min(),g.index.max(),freq='h'))
            origins=pd.date_range('2012-01-01','2013-12-31',freq='D')
            farms[int(farm)]=assemble(g.TARGETVAR,g,origins,year,int(farm),['U10','V10','U100','V100'])
    else:
        p=DATA/'gefcom2012_wind/data'
        source=p/'windpowermeasurements.csv';sources.append(source)
        power=pd.read_csv(source);power.index=pd.to_datetime(power.date.astype(str),format='%Y%m%d%H')
        assert power.index.equals(pd.date_range(power.index.min(),power.index.max(),freq='h'))
        for farm in range(1,8):
            path=p/f'windforecasts_wf{farm}.csv';sources.append(path)
            w=pd.read_csv(path);w['issue']=pd.to_datetime(w.date.astype(str),format='%Y%m%d%H')
            assert not w.duplicated(['date','hors']).any()
            groups={t:g.set_index('hors') for t,g in w.groupby('issue')}
            end=(w.issue+pd.to_timedelta(w.hors,unit='h')).max()-pd.Timedelta(hours=24)
            origins=pd.date_range(power.index.min().normalize(),end.normalize(),freq='D')
            weather={};ages={};issues={}
            for o in origins:
                for age in [0,12,24]:
                    t=o-pd.Timedelta(hours=age)
                    if t not in groups:continue
                    g=groups[t].reindex(np.arange(age+1,age+25))
                    arr=g[['u','v']].to_numpy(float)
                    if len(arr)==24 and np.isfinite(arr).all():
                        assert t<=o and age+24<=48
                        weather[o]=arr;ages[o]=age;issues[o]=t;break
            farms[farm]=assemble(power[f'wp{farm}'],weather,origins,year,farm,['u','v'],ages,issues)
    return farms,sources

def protocol(year,out,sources,policy=None):
    spec=dict(version=2,dataset=f'GEFCom{year}',model=PKG,quantiles=[.05,.95],
              history_hours=24,horizon_hours=24,origin_step_hours=24,capacity=1.,
              calibration='signed rolling CQR, 84 calendar days, per-lead; complete labels only after target_end <= origin',
              history='strict finite observed history; no history or target imputation',
              input_eligibility='history and selected complete NWP; independent of future label validity; same A/B/C calendar',
              target_quality='complete observed 24-hour labels for fitting/scoring; forecast invalid target days before inspecting arrival',
              zero_reporting_delay_assumption=True,nwp14_clock='inferred competition decision, not native ECMWF issue',
              nwp12_clock='latest complete native 00/12 batch not later than origin, lag 0/12/24h; its native leads 1+lag..24+lag',
              nwp12_height='single available wind field; not claimed to be the same height as GEFCom2014 100m',
              source_sha256={str(p):sha(p) for p in sources},python=sys.executable,
              versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__),
              runner_sha256=sha(Path(__file__)),calibration_sha256=sha(ROOT/'online_calibration.py'),
              methods=METHODS14 if year==2014 else METHODS12,
              representation_selection='GEFCom2014 Apr-Jun validation only: coverage>=.88, minimize macro IS; within1% relative IS choose fewer features; fallback highest coverage then IS',
              validation_seed='2013 Jan-Mar' if year==2014 else '2011 Jan-Mar',
              validation='2013 Apr-Jun' if year==2014 else '2011 Apr-Jun (diagnostic only; no reselection)',
              train='2012' if year==2014 else '2009 Jul-Dec and 2010',
              evaluation='2013 Jul-Dec (previously used, internal retrospective)' if year==2014 else '2011 Jul to 2012 Jun, frozen second-dataset comparison',
              evaluation_seed='2013 Apr-Jun' if year==2014 else '2011 Apr-Jun',frozen_transfer_policy=policy)
    path=out/'protocol.json'
    if path.exists():assert json.loads(path.read_text(encoding='utf-8'))==spec,'Protocol mismatch; use a new output version'
    else:write(path,spec)

def index(d,phase,valid=False):
    m=d['meta'].phase.eq(phase).to_numpy()
    return np.flatnonzero(m & d['valid'] if valid else m)

def run(year,out,farms):
    rows=[];leadrows=[];calchecks=[];started=time.perf_counter()
    methods=METHODS14 if year==2014 else METHODS12
    manifests=pd.concat([d['calendar'] for d in farms.values()],ignore_index=True)
    manifests.to_csv(out/'calendar_manifest.csv',index=False)
    manifests.groupby(['farm','phase']).agg(calendar_days=('origin','size'),issued=('input_valid','sum')).reset_index().to_csv(out/'service_counts.csv',index=False)
    for farm,d in farms.items():
        tr=index(d,'train',True);assert len(tr)>100
        vs=index(d,'validation_seed');vi=index(d,'validation');es=index(d,'validation');ei=index(d,'evaluation')
        assert d['ends'][tr].max()<=d['origins'][vs].min()
        assert d['ends'][vs].max()<=d['origins'][vi].min()
        assert d['ends'][es].max()<=d['origins'][ei].min()
        sample_path=out/f'farm_{farm:02d}';sample_path.mkdir(exist_ok=True)
        np.savez_compressed(sample_path/'samples.npz',y=d['y'],history=d['history'],nwp=d['nwp'],valid=d['valid'],origins=d['origins'],ends=d['ends'])
        d['meta'].to_csv(sample_path/'issued_manifest.csv',index=False)
        # Fit once. Evaluation metrics are not computed until representation is frozen.
        for method in methods:
            folder=sample_path/method;folder.mkdir(exist_ok=True);path=folder/'raw_predictions.npz'
            X=features(d,method,year);assert np.isfinite(X).all();t=time.perf_counter()
            if path.exists():
                with np.load(path) as z:lo=z['lo'];hi=z['hi'];assert np.array_equal(z['origins'],d['origins'])
            else:
                scaler=StandardScaler().fit(X[tr]);xs=scaler.transform(X);ps=[]
                joblib.dump(scaler,folder/'scaler.joblib')
                for q in [.05,.95]:
                    model=MultiOutputRegressor(HistGradientBoostingRegressor(loss='quantile',quantile=q,**PKG))
                    model.fit(xs[tr],d['y'][tr]);ps.append(model.predict(xs))
                    joblib.dump(model,folder/f'q{q:.2f}.joblib',compress=3)
                lo=np.minimum(*ps);hi=np.maximum(*ps)
                np.savez_compressed(path,lo=lo,hi=hi,origins=d['origins'])
            write(folder/'fit.json',dict(n_train=len(tr),features=X.shape[1],model=PKG,train_end=str(d['ends'][tr].max()),
                                        scoring_target_values_not_used_for_fit=True))
            # Validation uses only January-March seed and April-June labels arriving online.
            for stage,seed,score in [('validation',vs,vi)]:
                result=simulate(d['origins'],d['ends'],d['y'],lo,hi,d['valid'],seed,score,'rolling',84)
                save_cal(folder,stage,result,d)
                rows.extend(metric_rows(farm,method,stage,d,score,result['lo'],result['hi']))
            print(f'{year} farm{farm:02d} {method} fitted/validation features={X.shape[1]} secs={time.perf_counter()-t:.1f} elapsed={time.perf_counter()-started:.1f}',flush=True)
        pd.DataFrame(rows).to_csv(out/'validation_metrics_per_farm.csv',index=False)
    val=pd.DataFrame(rows);macro=macro_metrics(val);macro.to_csv(out/'validation_metrics_macro.csv',index=False)
    if year==2014:
        candidate=macro[(macro.group=='all')&macro.method.ne('A_history')].copy()
        eligible=candidate[candidate.coverage.ge(.88)]
        if len(eligible):
            best=eligible.IS.min();near=eligible[eligible.IS.le(best*1.01)].copy()
            dims={m:features(farms[1],m,year).shape[1] for m in methods}
            winner=min(near.method,key=lambda m:(dims[m],float(near.loc[near.method.eq(m),'IS'].iloc[0])))
        else:winner=candidate.sort_values(['coverage','IS'],ascending=[False,True]).iloc[0].method
        selected=dict(method=winner,representation='uv_plus_speed' if winner.endswith('speed') else 'uv',
                      selection_only='GEFCom2014 validation Apr-Jun; no July-Dec ranking',
                      selection_rule='minimum IS subject coverage >= .88; 1% near-best simplicity tie-break (not equivalence claim)',
                      candidate_metrics=candidate.to_dict('records'),
                      transfer_methods=METHODS12,transfer_choice='B_speed' if winner.endswith('speed') else 'B_uv',
                      shared_model=PKG,calibration_days=84,
                      warning='2014 height choice cannot be validated on 2012 single-level wind; latter validates encoding and NWP benefit')
        target=BASE/'frozen_transfer_policy.json'
        if target.exists():assert json.loads(target.read_text(encoding='utf-8'))==selected
        else:write(target,selected)
        print('FROZEN SELECTION',winner,'before evaluation metrics',flush=True)
    else:selected=json.loads((BASE/'frozen_transfer_policy.json').read_text(encoding='utf-8'))
    # Fixed retrospective evaluation. Both encoding controls retained without reselection.
    for farm,d in farms.items():
        seed=index(d,'validation');score=index(d,'evaluation')
        for method in methods:
            folder=out/f'farm_{farm:02d}'/method
            with np.load(folder/'raw_predictions.npz') as z:lo=z['lo'];hi=z['hi']
            result=simulate(d['origins'],d['ends'],d['y'],lo,hi,d['valid'],seed,score,'rolling',84)
            save_cal(folder,'evaluation',result,d)
            rows.extend(metric_rows(farm,method,'evaluation_cqr',d,score,result['lo'],result['hi']))
            rows.extend(metric_rows(farm,method,'evaluation_raw',d,score,lo[score],hi[score]))
            for h in range(24):
                met=metrics(d['y'][score,h:h+1],result['lo'][:,h:h+1],result['hi'][:,h:h+1],d['valid'][score],d['history'][score,-1])['all']
                leadrows.append(dict(farm=farm,method=method,lead=h+1,**met))
            calchecks.append(dict(farm=farm,method=method,issued=len(score),scored=int(d['valid'][score].sum()),
                              max_label_time_assertion=bool((result['max_label_end_used']<=d['origins'][score]).all()),
                              min_cal=int(result['n_cal'].min()),max_cal=int(result['n_cal'].max()),
                              negative_lower_cells=int((result['lo'][d['valid'][score]]<0).sum()),
                              empty_intervals=int(result['midpoint_expansion_count'])))
    allmetrics=pd.DataFrame(rows);allmetrics.to_csv(out/'metrics_per_farm.csv',index=False)
    macro_metrics(allmetrics).to_csv(out/'metrics_macro.csv',index=False)
    pd.DataFrame(leadrows).to_csv(out/'metrics_per_lead.csv',index=False)
    write(out/'verification.json',dict(checks=calchecks,first_hour_ramps_included=True,source_targets_unimputed=True,
           paired_origins=True,input_quality_independent_of_future_targets=True))
    write(out/'completion.json',dict(status='complete',farms=len(farms),methods=methods,elapsed_seconds=time.perf_counter()-started,
                                     selection=selected['method'] if year==2014 else selected['transfer_choice']))
    print('COMPLETE',year,flush=True)

def macro_metrics(frame):
    cols=['coverage','width','IS','lower_miss','upper_miss']
    return frame.groupby(['method','stage','group'])[cols].mean().reset_index()

def metric_rows(farm,method,stage,d,idx,lo,hi):
    met=metrics(d['y'][idx],lo,hi,d['valid'][idx],d['history'][idx,-1])
    return [dict(farm=farm,method=method,stage=stage,group=g,**v) for g,v in met.items()]

def save_cal(folder,stage,res,d):
    idx=res['indices'];assert (res['max_label_end_used']<=d['origins'][idx]).all()
    np.savez_compressed(folder/f'{stage}_intervals.npz',lo=res['lo'],hi=res['hi'],indices=idx,q=res['q_signed'],
                        n_cal=res['n_cal'],max_label_end=res['max_label_end_used'])
    pd.DataFrame(res['feedback']).to_csv(folder/f'{stage}_feedback.csv',index=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=int,choices=[2014,2012],required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    out=BASE/f'gefcom{a.dataset}';out.mkdir(parents=True,exist_ok=True)
    policy=None
    if a.dataset==2012:
        policy=json.loads((BASE/'frozen_transfer_policy.json').read_text(encoding='utf-8'))
    farms,sources=prepare(a.dataset);protocol(a.dataset,out,sources,policy)
    print('PREFLIGHT',a.dataset,'farms',len(farms),{m:features(farms[1],m,a.dataset).shape[1] for m in (METHODS14 if a.dataset==2014 else METHODS12)},flush=True)
    if not a.preflight_only:
        with threadpool_limits(limits=2):run(a.dataset,out,farms)

if __name__=='__main__':main()
