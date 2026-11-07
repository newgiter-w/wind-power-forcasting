"""Prespecified sequential tail-calibration/risk/dynamic-feature experiments."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,time,sys,platform,shutil
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from threadpoolctl import threadpool_limits
from gefcom_protocol_v2 import prepare,index,features,PKG
from strong_baselines import build_shared,order_quantile_bounds,_shift
from run_p2_2012 import build_shared_2012
from online_calibration import simulate,simulate_independent
from gefcom2014_experiment import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out_gefcom_tail_dynamic_v1'
REFERENCE=ROOT/'out_gefcom_factorial_20261103_v2'
METHODS=['R0_reference','C0_symmetric','C1_asymmetric','C2_risk','D01_base150','D10_dynamic50','D11_dynamic150']
RISK_PKG=dict(max_iter=50,max_leaf_nodes=7,min_samples_leaf=100,l2_regularization=1.,random_state=0,early_stopping=False)
GROUPS=['all','ramp','up_ramp','down_ramp','near_zero']
ROWS=[];CHECKS=[];OOF=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def dynamic_features(d):
    u=d['nwp'][:,:,-2];v=d['nwp'][:,:,-1];ws=np.hypot(u,v)
    values=[];names=[]
    for label,a in [('u',u),('v',v),('ws',ws)]:
        for lag in [1,3,6]:
            if label=='u' and lag==1:continue # already du_bwd in R2
            values.append(a-_shift(a,lag));names.append(f'{label}_delta{lag}')
    safe=np.maximum(ws,1e-6)
    for label,a in [('dir_u',np.where(ws>1e-6,u/safe,0)),('dir_v',np.where(ws>1e-6,v/safe,0))]:
        values.append(a-_shift(a,1));names.append(label+'_delta1')
    for label,a in [('u',u),('v',v),('ws',ws)]:
        values.append(a-a[:,:1]);names.append(label+'_minus_h1')
    hist=d['history'];shape=ws.shape
    values.extend([np.broadcast_to(hist[:,-6:].std(axis=1)[:,None],shape),
        np.broadcast_to((hist[:,-1]-hist[:,-6])[:,None],shape)])
    names.extend(['past6_std','past6_change'])
    values.extend([(ws-_shift(ws,3))*hist[:,-1,None],
        (ws-ws[:,:1])*hist[:,-1,None]*np.exp(-np.arange(1,25)[None,:]/6)])
    names.extend(['ws_delta3_x_last_power','ws_from_h1_x_last_power_decay6'])
    assert len(values)==17
    return np.stack(values,axis=2),names

def base_matrix(d,year):
    base=features(d,'B_speed',year);S=len(base)
    old,y,_=(build_shared if year==2014 else build_shared_2012)(d,'R2')
    np.testing.assert_equal(y.reshape(S,24),d['y'])
    extra=old[:,-14:]
    X=np.c_[np.repeat(base,24,axis=0),np.tile(np.arange(1,25),S),extra]
    return np.ascontiguousarray(X)

def fit(X,y,train,q,pkg):
    X=np.ascontiguousarray(X);keep=~np.isnan(X[train]).all(axis=0)
    X=np.ascontiguousarray(X[:,keep]);sc=StandardScaler().fit(X[train]);xx=sc.transform(X)
    model=HistGradientBoostingRegressor(loss='quantile',quantile=q,**pkg).fit(xx[train],y[train])
    return model.predict(xx),dict(model=model,scaler=sc,keep=keep)

def predict(pack,X):
    return pack['model'].predict(pack['scaler'].transform(np.ascontiguousarray(X[:,pack['keep']])))

def fit_pair(X,y,train,pkg,folder):
    p=folder/'heads.npz';folder.mkdir(parents=True,exist_ok=True)
    if p.exists():
        with np.load(p) as z:
            np.testing.assert_array_equal(z['train_rows'],np.flatnonzero(train))
            return z['q05'],z['q95']
    a,ma=fit(X,y,train,.05,pkg);b,mb=fit(X,y,train,.95,pkg)
    joblib.dump((ma,mb),folder/'models.joblib',compress=3)
    np.savez_compressed(p,q05=a,q95=b,train_rows=np.flatnonzero(train))
    return a,b

def folder_for(year,farm):return OUT/f'gefcom{year}'/f'farm_{farm:02d}'

def frozen_protocol():
    OUT.mkdir(exist_ok=True)
    dependencies=['run_gefcom_tail_dynamic.py','gefcom_protocol_v2.py','online_calibration.py',
        'strong_baselines.py','run_p2_2012.py','gefcom2014_experiment.py','test_gefcom_tail_dynamic.py']
    spec=dict(date='2026-11-03',methods=METHODS,stages=['2014_validation_C1','2014_validation_C2',
        '2014_validation_dynamic','freeze_selection','2014_evaluation_fixed_ablations','2012_frozen_transfer'],
        base_model=PKG,risk_model=RISK_PKG,iterations_dynamic=[50,150],new_dynamic_columns=17,
        calibration='84 calendar days per lead; symmetric90% or independent95% tails; signed corrections; clip[0,1] last',
        time_splits='unchanged protocol_v2; no Q1 fitting; OOF only proper-training Q3/Q4',
        oof='expanding proper-training prefixes; assert maximum arrived training-label end<=first held-out origin; no Q1 auxiliary training',
        selection='2014Q2 only; all point gates; among eligible minIS then within1% simplest; no eligible=>C0 unchanged',
        gates=dict(coverage_min=.90,coverage_drop_max=.005,width_relative_reduction=.05,IS_relative_reduction=.05,
            up_coverage_gain=.03,up_IS_nonincrease=True,up_width_increase_max=.02,
            near_zero_min='max(R0-.01,C0-.005)'),
        simplicity_order=['C1_asymmetric','D10_dynamic50','D01_base150','D11_dynamic150','C2_risk'],
        evaluation='all seven prespecified references/ablations; no combination search; no evaluation tuning',
        limits=['fixed farms and seed0; previously inspected dates are retrospective','no conditional coverage theorem under time dependence'],
        bootstrap='analysis: paired2000 moving7/14-day common calendar draws across farms/methods',
        plan_sha256=sha(ROOT/'refine-logs/EXPERIMENT_PLAN_20261103_123419.md'),
        reference_protocol_sha256=sha(REFERENCE/'protocol.json'),
        source_sha256={n:sha(ROOT/n) for n in dependencies},
        python=sys.executable,versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__))
    p=OUT/'protocol.json'
    if p.exists():assert json.loads(p.read_text(encoding='utf-8'))==spec,'Frozen protocol changed: use a new version'
    else:
        write(p,spec)
        snap=OUT/'source_snapshot';snap.mkdir(exist_ok=True)
        for n in dependencies:shutil.copy2(ROOT/n,snap/n)
    return spec

def save_progress():
    frame=pd.DataFrame(ROWS)
    frame.to_csv(OUT/'metrics_per_farm.csv',index=False)
    frame.groupby(['year','method','stage','kind','support','group'])[['coverage','width','IS','lower_miss','upper_miss']].mean().reset_index().to_csv(OUT/'metrics_macro.csv',index=False)
    write(OUT/'checks.json',CHECKS);write(OUT/'oof_checks.json',OOF)

def record(year,farm,d,method,lo,hi,stage,independent):
    target=folder_for(year,farm)/method;target.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target/'raw.npz',lo=lo,hi=hi,origins=d['origins'])
    seed=index(d,'validation_seed' if stage=='validation' else 'validation');score=index(d,stage)
    fn=simulate_independent if independent else simulate
    result=fn(d['origins'],d['ends'],d['y'],lo,hi,d['valid'],seed,score,'rolling',84)
    assert (result['max_label_end_used']<=d['origins'][score]).all()
    assert np.isfinite(result['lo']).all() and np.isfinite(result['hi']).all()
    assert (result['lo']<=result['hi']).all()
    if method in ['R0_reference','C0_symmetric']:
        ref=REFERENCE/f'gefcom{year}'/f'farm_{farm:02d}'/('R0_ind_base' if method=='R0_reference' else 'R2_shared_aug')/f'{stage}.npz'
        with np.load(ref) as z:
            np.testing.assert_equal(z['indices'],score)
            np.testing.assert_array_equal(z['lo'],result['lo']);np.testing.assert_array_equal(z['hi'],result['hi'])
    arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
    arrays['max_label_end']=result['max_label_end_used']
    np.savez_compressed(target/f'{stage}.npz',**arrays)
    pd.DataFrame(result['feedback']).to_csv(target/f'{stage}_feedback.csv',index=False)
    write(target/f'{stage}_calibration.json',{k:v for k,v in result.items() if not isinstance(v,np.ndarray) and k!='feedback'})
    for kind,L,U in [('raw',lo[score],hi[score]),('cqr',result['lo'],result['hi'])]:
        for support in ['unclipped','clipped']:
            l,u=(np.clip(L,0,1),np.clip(U,0,1)) if support=='clipped' else (L,U)
            for g,v in metrics(d['y'][score],l,u,d['valid'][score],d['history'][score,-1]).items():
                ROWS.append(dict(year=year,farm=farm,method=method,stage=stage,kind=kind,support=support,group=g,**v))
    CHECKS.append(dict(year=year,farm=farm,method=method,stage=stage,issued_days=len(score),scored_days=int(d['valid'][score].sum()),
        causal_feedback=True,min_calibration_days=int(result['n_cal'].min()),
        empty_interval_protections=int(result['midpoint_expansion_count'])))
    save_progress()

def load_data(year):
    farms,sources=prepare(year)
    digests={str(p):sha(p) for p in sources}
    assert digests==json.loads((REFERENCE/f'gefcom{year}/raw_sources.json').read_text(encoding='utf-8'))
    write(OUT/f'gefcom{year}/raw_sources.json',digests)
    for farm,d in farms.items():
        folder=folder_for(year,farm);folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'samples.npz',**{k:d[k] for k in ['y','history','nwp','valid','origins','ends']})
        d['meta'].to_csv(folder/'issued_manifest.csv',index=False)
        d['calendar'].to_csv(folder/'calendar.csv',index=False)
        ref=REFERENCE/f'gefcom{year}'/f'farm_{farm:02d}'
        with np.load(ref/'samples.npz') as z:
            for key in ['y','history','valid','origins','ends']:np.testing.assert_equal(d[key],z[key])
        X=base_matrix(d,year);S=len(d['y']);d['_X']=X
        packs=joblib.load(ref/'R2_shared_aug/models.joblib')
        with np.load(ref/'R2_shared_aug/heads.npz') as z:
            np.testing.assert_array_equal(predict(packs[0],X).reshape(S,24),z['q05'])
            np.testing.assert_array_equal(predict(packs[1],X).reshape(S,24),z['q95'])
            lo,hi=order_quantile_bounds(z['q05'],z['q95'])
        d['_raw']={'C0_symmetric':(lo,hi),'C1_asymmetric':(lo,hi)}
        with np.load(ref/'R0_ind_base/heads.npz') as z:d['_raw']['R0_reference']=order_quantile_bounds(z['q05'],z['q95'])
        write(folder/'reference_identity.json',dict(samples_exact=True,R2_model_predictions_exact=True,
            model_sha256=sha(ref/'R2_shared_aug/models.joblib'),heads_sha256=sha(ref/'R2_shared_aug/heads.npz')))
    return farms

def risk_matrix(d,proxy,lo,hi):
    u=d['nwp'][:,:,-2];v=d['nwp'][:,:,-1];ws=np.hypot(u,v);hist=d['history'];S=len(u)
    p=proxy.predict(ws.ravel()).reshape(S,24)
    fields=[np.broadcast_to(np.arange(1,25),(S,24)),
        np.broadcast_to(np.sin(2*np.pi*np.arange(1,25)/24),(S,24)),
        np.broadcast_to(np.cos(2*np.pi*np.arange(1,25)/24),(S,24)),u,v,ws]
    names=['lead','hour_sin','hour_cos','u','v','ws']
    for label,a in [('u',u),('v',v),('ws',ws)]:
        for lag in [1,3]:fields.append(a-_shift(a,lag));names.append(f'{label}_delta{lag}')
    fields.extend([p,np.diff(p,axis=1,prepend=hist[:,-1,None]),p-hist[:,-1,None],
        np.broadcast_to(hist[:,-1,None],p.shape),np.broadcast_to(hist[:,-6:].std(axis=1)[:,None],p.shape),lo,hi,hi-lo])
    names.extend(['proxy_power','proxy_change','proxy_minus_last','last_power','past6_std','raw_lo','raw_hi','raw_width'])
    day=pd.DatetimeIndex(d['origins']).dayofyear.to_numpy()
    fields.extend([np.broadcast_to(np.sin(2*np.pi*day/365.25)[:,None],p.shape),np.broadcast_to(np.cos(2*np.pi*day/365.25)[:,None],p.shape)])
    names.extend(['season_sin','season_cos'])
    return np.ascontiguousarray(np.stack(fields,axis=2).reshape(S*24,-1)),names

def fit_proxy(d,tr):
    ws=np.hypot(d['nwp'][:,:,-2],d['nwp'][:,:,-1])
    return IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(ws[tr].ravel(),d['y'][tr].ravel())

def oof_splits(d,year):
    yr=2012 if year==2014 else 2010
    proper=index(d,'train',True);orig=d['origins'];ends=d['ends']
    splits=[]
    for quarter,start,end in [('Q3',f'{yr}-07-01',f'{yr}-10-01'),('Q4',f'{yr}-10-01',f'{yr+1}-01-01')]:
        held=proper[(orig[proper]>=np.datetime64(start))&(orig[proper]<np.datetime64(end))]
        train=proper[(orig[proper]<np.datetime64(start))&(ends[proper]<=orig[held].min())]
        assert len(train)>30 and len(held)>30 and not np.intersect1d(train,held).size
        assert ends[train].max()<=orig[held].min()
        splits.append((quarter,train,held))
    return splits

def train_risk(year,farm,d):
    folder=folder_for(year,farm)/'C2_risk';folder.mkdir(parents=True,exist_ok=True)
    full=folder/'adjustment.npz'
    # Rebuild OOF manifests even when cached fits are reused.
    matrices=[];low=[];up=[];witnesses=[]
    for quarter,tr,held in oof_splits(d,year):
        part=folder/'oof'/quarter;mask=np.repeat(np.isin(np.arange(len(d['y'])),tr),24)
        a,b=fit_pair(d['_X'],d['y'].ravel(),mask,PKG,part)
        lo,hi=order_quantile_bounds(a.reshape(-1,24),b.reshape(-1,24))
        proxy=fit_proxy(d,tr);joblib.dump(proxy,part/'proxy.joblib')
        X,names=risk_matrix(d,proxy,lo,hi)
        sel=(held[:,None]*24+np.arange(24)).ravel()
        matrices.append(X[sel]);low.append((lo-d['y'])[held].ravel());up.append((d['y']-hi)[held].ravel())
        witness=dict(year=year,farm=farm,quarter=quarter,train_days=len(tr),heldout_days=len(held),
            max_train_label_end=str(d['ends'][tr].max()),first_prediction_origin=str(d['origins'][held].min()),
            last_held_label_end=str(d['ends'][held].max()),no_overlap=True,arrived_labels_only=True)
        OOF.append(witness);witnesses.append(witness)
        np.savez_compressed(part/'residuals.npz',indices=held,train_indices=tr,X=X[sel],
            r_lower=low[-1],r_upper=up[-1],lo=lo[held],hi=hi[held],origins=d['origins'][held])
    Xtr=np.ascontiguousarray(np.concatenate(matrices));yl=np.concatenate(low);yu=np.concatenate(up)
    tr=index(d,'train',True);proxy=fit_proxy(d,tr);joblib.dump(proxy,folder/'final_proxy.joblib')
    lo,hi=d['_raw']['C0_symmetric'];X,names=risk_matrix(d,proxy,lo,hi)
    assert d['ends'][tr].max()<=d['origins'][index(d,'validation_seed')].min()
    # Fit only the concatenated OOF matrices, then predict the separate final-forecaster matrix.
    if full.exists():
        with np.load(full) as z:bl=z['b_lower'];bu=z['b_upper']
    else:
        _,ml=fit(Xtr,yl,np.arange(len(yl)),.95,RISK_PKG)
        _,mu=fit(Xtr,yu,np.arange(len(yu)),.95,RISK_PKG)
        joblib.dump((ml,mu),folder/'risk_models.joblib',compress=3)
        bl=predict(ml,X).reshape(-1,24);bu=predict(mu,X).reshape(-1,24)
        np.savez_compressed(full,b_lower=bl,b_upper=bu,origins=d['origins'])
    L=lo-bl;U=hi+bu;ordered=order_quantile_bounds(L,U)
    d['_raw']['C2_risk']=ordered
    write(folder/'fit.json',dict(features=names,OOF=witnesses,risk_rows=len(yl),effective_heldout_days=len(yl)//24,
        final_proper_train_days=len(tr),Q1_used_for_fitting=False,auxiliary_labels_latest=max(x['last_held_label_end'] for x in witnesses),
        adjusted_crossings=int((L>U).sum()),b_lower_quantiles=np.quantile(bl,[0,.05,.5,.95,1]).tolist(),
        b_upper_quantiles=np.quantile(bu,[0,.05,.5,.95,1]).tolist()))

def train_dynamic(year,farm,d,method):
    extra,names=dynamic_features(d);dynamic='dynamic' in method;iters=150 if method.endswith('150') else 50
    X=np.c_[d['_X'],extra.reshape(-1,17)] if dynamic else d['_X']
    tr=index(d,'train',True);mask=np.repeat(np.isin(np.arange(len(d['y'])),tr),24)
    p=dict(PKG,max_iter=iters);folder=folder_for(year,farm)/method
    a,b=fit_pair(X,d['y'].ravel(),mask,p,folder)
    d['_raw'][method]=order_quantile_bounds(a.reshape(-1,24),b.reshape(-1,24))
    write(folder/'fit.json',dict(model=p,features=X.shape[1],new_columns=names if dynamic else [],
        train_days=len(tr),max_training_label_end=str(d['ends'][tr].max()),same_full_training_samples=True,
        crossing_cells=int((a>b).sum())))

def freeze_selection():
    f=pd.read_csv(OUT/'metrics_macro.csv')
    f=f[(f.year==2014)&(f.stage=='validation')&(f.kind=='cqr')&(f.support=='clipped')]
    def get(m,g):return f[(f.method==m)&(f.group==g)].iloc[0]
    base=get('C0_symmetric','all');bup=get('C0_symmetric','up_ramp');bnz=get('C0_symmetric','near_zero')
    rnz=get('R0_reference','near_zero');records=[]
    simplicity=['C1_asymmetric','D10_dynamic50','D01_base150','D11_dynamic150','C2_risk']
    for method in simplicity:
        a=get(method,'all');up=get(method,'up_ramp');nz=get(method,'near_zero')
        gates=dict(coverage90=bool(a.coverage>=.9),coverage_noninferior_point=bool(a.coverage>=base.coverage-.005),
            width_5pct=bool(a.width<=.95*base.width),IS_5pct=bool(a.IS<=.95*base.IS),
            up_coverage_3pp=bool(up.coverage>=bup.coverage+.03),up_IS=bool(up.IS<=bup.IS),
            up_width=bool(up.width<=bup.width+.02),near_zero=bool(nz.coverage>=max(rnz.coverage-.01,bnz.coverage-.005)))
        records.append(dict(method=method,IS=float(a.IS),coverage=float(a.coverage),width=float(a.width),
            up_coverage=float(up.coverage),near_zero_coverage=float(nz.coverage),gates=gates,eligible=all(gates.values())))
    eligible=[r for r in records if r['eligible']]
    if eligible:
        best=min(r['IS'] for r in eligible)
        choice=min((r for r in eligible if r['IS']<=1.01*best),key=lambda r:simplicity.index(r['method']))['method']
    else:choice='C0_symmetric'
    selection=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),validation='2014Q2 only',
        choice=choice,promoted=bool(eligible),candidates=records,
        reason='all point gates then IS/simplicity' if eligible else 'no candidate meets all prespecified point gates; retain C0',
        no_evaluation_metrics_read=True,validation_macro_sha256=sha(OUT/'metrics_macro.csv'))
    p=OUT/'selection.json'
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'));assert old['choice']==choice and old['candidates']==records
    else:write(p,selection)
    f.to_csv(OUT/'frozen_validation_macro.csv',index=False)
    print('SELECTION FROZEN',choice,'promoted=',bool(eligible),flush=True)

def validation(farms):
    year=2014
    for method in ['R0_reference','C0_symmetric','C1_asymmetric']:
        for farm,d in farms.items():
            record(year,farm,d,method,*d['_raw'][method],'validation',method=='C1_asymmetric')
        print('VALIDATION STAGE COMPLETE',method,flush=True)
    for farm,d in farms.items():
        train_risk(year,farm,d);record(year,farm,d,'C2_risk',*d['_raw']['C2_risk'],'validation',True)
        print('VALIDATION C2 farm',farm,flush=True)
    for method in ['D01_base150','D10_dynamic50','D11_dynamic150']:
        for farm,d in farms.items():
            train_dynamic(year,farm,d,method);record(year,farm,d,method,*d['_raw'][method],'validation',False)
        print('VALIDATION STAGE COMPLETE',method,flush=True)

def main():
    started=time.perf_counter();frozen_protocol()
    farms=load_data(2014);validation(farms);freeze_selection()
    for farm,d in farms.items():
        for method in METHODS:record(2014,farm,d,method,*d['_raw'][method],'evaluation',method in ['C1_asymmetric','C2_risk'])
        print('2014 EVALUATION farm',farm,flush=True)
    farms=load_data(2012)
    for farm,d in farms.items():
        train_risk(2012,farm,d)
        for method in ['D01_base150','D10_dynamic50','D11_dynamic150']:train_dynamic(2012,farm,d,method)
        for stage in ['validation','evaluation']:
            for method in METHODS:record(2012,farm,d,method,*d['_raw'][method],stage,method in ['C1_asymmetric','C2_risk'])
        print('2012 FROZEN TRANSFER farm',farm,flush=True)
    write(OUT/'completion.json',dict(status='complete',farm_method_groups=119,stage_groups=len(CHECKS),
        OOF_blocks=len(OOF),elapsed_seconds=time.perf_counter()-started,selection_sha256=sha(OUT/'selection.json')))
    print('COMPLETE elapsed',time.perf_counter()-started,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
