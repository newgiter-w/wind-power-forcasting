"""Reviewer arithmetic from saved arrays. No experiment functions are imported."""
from pathlib import Path
import json, csv, hashlib
import numpy as np
import pandas as pd

ROOT=Path('D:/Study/wind_probability_route_a')
RUN=ROOT/'experiments/route_a/20261107_p1_v030'
P0=ROOT/'experiments/route_a/20261107_p0_v020'
OUT=RUN/'review_trace'
ARMS=['QRF_linear_raw','CQR_signed84','DualTail84','QRF_ECDF_raw','PIT84','PIT84_state','DualTail84_state']
Q=np.array([.025,.05,.1,.5,.9,.95,.975])
MKEY=['year','farm','stage','method','confidence','dimension','group']
SKEY=['year','farm','stage','method','group']
saved_m=pd.read_csv(RUN/'metrics_per_farm.csv',dtype={'group':str}).set_index(MKEY)
saved_s=pd.read_csv(RUN/'scores_per_farm.csv',dtype={'group':str}).set_index(SKEY)
saved_b=pd.read_csv(RUN/'boundary_metrics.csv').set_index(['year','farm','stage','method'])
saved_st=pd.read_csv(RUN/'state_support.csv').set_index(['year','farm','stage','state'])
errors={}; counts={}; daily={}; rec_m=[];rec_s=[];calendars={};nwp_gaps={}
def check(name,actual,expected,atol=2e-12):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    np.testing.assert_allclose(a,b,rtol=0,atol=atol,equal_nan=True,err_msg=name)
    dif=np.abs(a-b);err=float(np.nanmax(dif)) if np.any(np.isfinite(dif)) else 0.
    errors[name]=max(errors.get(name,0.),err);counts[name]=counts.get(name,0)+a.size
def interval(y,lo,hi,alpha):
    below=y<lo;above=y>hi
    return {'coverage':~(below|above),'width':hi-lo,'IS':hi-lo+np.where(below,2/alpha*(lo-y),0)+np.where(above,2/alpha*(y-hi),0),'lower_miss':below,'upper_miss':above}
for year,nfarm in [(2014,10),(2012,7)]:
    c=pd.read_csv(P0/f'gefcom{year}/farm_01/calendar.csv')
    dates=pd.to_datetime(c.loc[c.phase.eq('evaluation'),'origin']).to_numpy()
    assert np.array_equal(dates,np.arange(dates[0],dates[-1]+np.timedelta64(1,'D'),np.timedelta64(1,'D')))
    calendars[year]=dates;nwp_gaps[year]=[]
    for farm in range(1,nfarm+1):
        prior=P0/f'gefcom{year}/farm_{farm:02d}';folder=RUN/f'gefcom{year}/farm_{farm:02d}'
        with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in z.files}
        assert np.array_equal(d['valid'],np.isfinite(d['y']).all(axis=1)&(d['y']>=0).all(axis=1)&(d['y']<=1).all(axis=1))
        meta=pd.read_csv(prior/'samples.csv')
        c2=pd.read_csv(prior/'calendar.csv');dates2=pd.to_datetime(c2.loc[c2.phase.eq('evaluation'),'origin']).to_numpy();assert np.array_equal(dates,dates2)
        with np.load(folder/'distribution_inputs.npz') as z:states=z['states']
        for stage in ['validation','evaluation']:
            with np.load(folder/f'{stage}.npz') as z:pred=z['pred'];ids=z['indices'];crps=z['crps'];local_days=z['local_days'];lambdas=z['lambdas'];methods=z['methods']
            assert list(methods)==ARMS
            np.testing.assert_array_equal(ids,np.flatnonzero(meta.phase.eq(stage)))
            y=d['y'][ids];valid=np.broadcast_to(d['valid'][ids,None],y.shape)
            assert np.isfinite(pred).all()
            assert np.isnan(crps[[0,1,2,6]]).all()
            if stage=='evaluation':nwp_gaps[year].append(len(dates)-len(ids))
            delta=np.concatenate([y[:,0,None]-d['history'][ids,-1,None],y[:,1:]-y[:,:-1]],axis=1)
            masks=[('regime','all',np.ones_like(y,dtype=bool)),('regime','up_ramp',delta>.1),('regime','down_ramp',delta<-.1),('regime','near_zero',y<=.01)]
            for low,high,label in [(0,.01,'0_to_.01'),(.01,.2,'.01_to_.2'),(.2,.8,'.2_to_.8'),(.8,1,'.8_to_1')]:masks.append(('power',label,((y>=low) if low==0 else (y>low))&(y<=high)))
            for h in range(24):masks.append(('lead',str(h+1),np.broadcast_to(np.arange(24)==h,y.shape)))
            timestamps=d['origins'][ids,None]+np.arange(1,25)[None,:]*np.timedelta64(1,'h')
            months=timestamps.astype('datetime64[M]').astype(str)
            for month in np.unique(months):masks.append(('month',month,months==month))
            mn=pd.DatetimeIndex(timestamps.ravel()).month.to_numpy().reshape(y.shape)
            for season,mns in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:masks.append(('season',season,np.isin(mn,mns)))
            for arm,method in enumerate(ARMS):
                wis=.5*abs(y-pred[arm,:,:,3]);common=.5*abs(y-pred[0,:,:,3])
                for alpha,li,ui in [(.2,2,4),(.1,1,5),(.05,0,6)]:
                    stats=interval(y,pred[arm,:,:,li],pred[arm,:,:,ui],alpha)
                    wis+=alpha/2*stats['IS'];common+=alpha/2*stats['IS']
                    for dim,group,mask in masks:
                        sel=valid&mask
                        if not sel.any():continue
                        values={'cells':int(sel.sum()),'days':int(sel.any(axis=1).sum()),**{k:float(v[sel].mean()) for k,v in stats.items()}}
                        old=saved_m.loc[year,farm,stage,method,1-alpha,dim,group]
                        check('interval_metrics',list(values.values()),old[list(values)].to_numpy())
                        rec_m.append(dict(year=year,farm=farm,stage=stage,method=method,confidence=1-alpha,dimension=dim,group=group,**values))
                wis/=3.5;common/=3.5
                err=y[:,:,None]-pred[arm];pin=np.where(err>=0,Q*err,(Q-1)*err).mean(axis=-1)
                check('WIS_pinball_identity',wis[valid],2*pin[valid])
                stat90=interval(y,pred[arm,:,:,1],pred[arm,:,:,5],.1)
                for dim,group,mask in masks[:4]:
                    sel=valid&mask
                    if not sel.any():continue
                    values=dict(cells=int(sel.sum()),days=int(sel.any(axis=1).sum()),WIS=float(wis[sel].mean()),common_median_WIS=float(common[sel].mean()),pinball7=float(pin[sel].mean()),median_MAE=float(abs(y-pred[arm,:,:,3])[sel].mean()),CRPS=float(crps[arm][sel].mean()) if arm in (3,4,5) else np.nan,nested_80_90=int((((pred[arm,:,:,2]<pred[arm,:,:,1])|(pred[arm,:,:,4]>pred[arm,:,:,5]))&sel).sum()),nested_90_95=int((((pred[arm,:,:,1]<pred[arm,:,:,0])|(pred[arm,:,:,5]>pred[arm,:,:,6]))&sel).sum()))
                    old=saved_s.loc[year,farm,stage,method,group]
                    check('scores',list(values.values()),old[list(values)].to_numpy())
                    rec_s.append(dict(year=year,farm=farm,stage=stage,method=method,group=group,**values))
                    if stage=='evaluation' and group!='down_ramp':
                        stats={**stat90,'WIS':wis}
                        if arm in (3,4,5):stats['CRPS']=crps[arm]
                        daily[year,farm,method,group]=(d['origins'][ids],sel.sum(axis=1),{k:np.where(sel,v,0).sum(axis=1) for k,v in stats.items()})
            with np.load(folder/f'{stage}_boundary_probabilities.npz') as z:
                np.testing.assert_array_equal(z['indices'],ids)
                for method in ARMS[3:6]:
                    probs=z[method];event=(y<=.01)
                    vals=dict(cells=int(valid.sum()),near_zero_Brier=float(((probs[:,:,0]-event)**2)[valid].mean()),mean_predicted_near_zero=float(probs[:,:,0][valid].mean()),observed_near_zero=float(event[valid].mean()),mean_zero_mass=float(probs[:,:,1][valid].mean()),mean_one_mass=float(probs[:,:,2][valid].mean()))
                    check('boundary_score_aggregation',list(vals.values()),saved_b.loc[year,farm,stage,method][list(vals)].to_numpy())
            for state,label in [(0,'predicted_near_zero'),(1,'predicted_up_ramp'),(2,'other')]:
                sel=valid&(states[ids]==state)
                vals=dict(cells=int(sel.sum()),all_cells=int(valid.sum()),local_fallback_fraction=float((lambdas[sel]==0).mean()) if sel.any() else np.nan,mean_lambda=float(lambdas[sel].mean()) if sel.any() else np.nan,mean_local_distinct_days=float(local_days[sel].mean()) if sel.any() else np.nan)
                check('state_support',list(vals.values()),saved_st.loc[year,farm,stage,label][list(vals)].to_numpy())
frame=pd.DataFrame(rec_m);ss=pd.DataFrame(rec_s)
mk=['year','stage','method','confidence','dimension','group'];sk=['year','stage','method','group']
macro=frame.groupby(mk).agg(farms=('farm','nunique'),cells=('cells','sum'),**{k:(k,'mean') for k in ['coverage','width','IS','lower_miss','upper_miss']})
smacro=ss.groupby(sk).agg(farms=('farm','nunique'),cells=('cells','sum'),**{k:(k,'mean') for k in ['WIS','common_median_WIS','pinball7','median_MAE','CRPS']},nested_80_90=('nested_80_90','sum'),nested_90_95=('nested_90_95','sum'))
check('macro_metrics',macro.sort_index(),pd.read_csv(RUN/'metrics_macro.csv',dtype={'group':str}).set_index(mk)[macro.columns].sort_index())
check('macro_scores',smacro.sort_index(),pd.read_csv(RUN/'scores_macro.csv').set_index(sk)[smacro.columns].sort_index())
assert len(frame)==len(saved_m) and len(ss)==len(saved_s)
# Recalculate every reported paired interval with multiplicity counts per draw,
# using independently reconstructed daily arrays rather than saved sufficient sums.
boot=pd.read_csv(RUN/'paired_calendar_blocks.csv').set_index(['year','group','method','reference','metric'])
rng=np.random.default_rng(20261107)
for year,nfarms in [(2014,10),(2012,7)]:
    dates=calendars[year];n=len(dates)
    starts=rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1))
    sampled=((starts+np.arange(14))%n).reshape(1000,-1)[:,:n]
    multiplicities=np.zeros((1000,n),dtype=np.int32)
    for draw in range(1000):multiplicities[draw]=np.bincount(sampled[draw],minlength=n)
    for group in ['all','up_ramp','near_zero']:
        estimates={};points={}
        for method in ARMS:
            for metric in ['coverage','width','IS','WIS','CRPS']:
                if metric=='CRPS' and method not in ARMS[3:6]:continue
                farmdraw=[];farmpoint=[]
                for farm in range(1,nfarms+1):
                    origins,counts1,stats=daily[year,farm,method,group]
                    positions=np.searchsorted(dates,origins);assert np.array_equal(dates[positions],origins)
                    sums=np.zeros(n);counts1a=np.zeros(n);sums[positions]=stats[metric];counts1a[positions]=counts1
                    den=multiplicities@counts1a;assert (den>0).all()
                    farmdraw.append((multiplicities@sums)/den);farmpoint.append(sums.sum()/counts1a.sum())
                estimates[method,metric]=np.mean(farmdraw,axis=0);points[method,metric]=np.mean(farmpoint)
        for (yr,g,method,reference,metric),old in boot.loc[[year]].iterrows():
            if g!=group:continue
            dif=estimates[method,metric]-estimates[reference,metric]
            vals=[points[method,metric]-points[reference,metric],*np.quantile(dif,[.025,.975])]
            check('paired_calendar_bootstrap',vals,old[['delta','ci_low','ci_high']].to_numpy())
# Independent predicates against reconstructed macro data.
gates=pd.read_csv(RUN/'joint_gates.csv').set_index(['year','stage','method'])
gate_rows=[]
for (year,stage,method),old in gates.iterrows():
    a=macro.loc[year,stage,method,.9,'regime','all'];b=macro.loc[year,stage,'CQR_signed84',.9,'regime','all']
    up=macro.loc[year,stage,method,.9,'regime','up_ramp'];bu=macro.loc[year,stage,'CQR_signed84',.9,'regime','up_ramp']
    zero=macro.loc[year,stage,method,.9,'regime','near_zero'];bz=macro.loc[year,stage,'CQR_signed84',.9,'regime','near_zero']
    s=smacro.loc[year,stage,method,'all'];bs=smacro.loc[year,stage,'CQR_signed84','all']
    passes=dict(coverage=a.coverage>=.9,coverage_drop=a.coverage>=b.coverage-.005,width=a.width<=b.width*1.01,IS=a.IS<=b.IS,WIS=s.WIS<=bs.WIS,up_coverage=up.coverage>=bu.coverage+.01,zero_coverage=zero.coverage>=bz.coverage+.01,up_IS=up.IS<=bu.IS,zero_IS=zero.IS<=bz.IS)
    vals=dict(coverage=a.coverage,width_ratio=a.width/b.width,IS_ratio=a.IS/b.IS,WIS_ratio=s.WIS/bs.WIS,common_median_WIS_ratio=s.common_median_WIS/bs.common_median_WIS,up_coverage_gain=up.coverage-bu.coverage,zero_coverage_gain=zero.coverage-bz.coverage,up_IS_ratio=up.IS/bu.IS,zero_IS_ratio=zero.IS/bz.IS)
    check('gate_numbers',list(vals.values()),old[list(vals)].to_numpy())
    for name,value in passes.items():assert bool(old['pass_'+name])==bool(value)
    assert bool(old.joint_pass)==all(passes.values())
    eligible=method in ['DualTail84','PIT84','PIT84_state','DualTail84_state'];assert bool(old.eligible)==eligible
    gate_rows.append(dict(year=year,stage=stage,method=method,eligible=eligible,joint_pass=all(passes.values()),**vals))
dev=[x for x in gate_rows if x['year']==2014 and x['stage']=='validation' and x['eligible'] and x['joint_pass']]
selected=min(dev,key=lambda x:(x['WIS_ratio'],x['method']))['method'] if dev else 'CQR_signed84'
saved_selection=json.loads((RUN/'selection.json').read_text())
assert selected==saved_selection['selected'] and [x['method'] for x in dev]==saved_selection['eligible_passing'] and saved_selection['evaluation_used_for_selection'] is False
# Check all four raw reproduction metric fields against P0 across every group.
pm=pd.read_csv(P0/'metrics_macro.csv',dtype={'group':str})
for old,new in [('QRF_raw','QRF_linear_raw'),('QRF_CQR_rolling84','CQR_signed84')]:
    cols=['coverage','width','IS','lower_miss','upper_miss'];keys=['year','stage','confidence','dimension','group']
    check('P0_controls',pm[pm.method.eq(old)].set_index(keys)[cols].sort_index(),macro.reset_index().query('method==@new').set_index(keys)[cols].sort_index())
evidence=json.loads((RUN/'evidence_values.json').read_text())
with (RUN/'joint_gates.csv').open(newline='') as f:rawg=list(csv.DictReader(f))
for row in evidence['rows']:assert row in rawg
result=dict(status='pass',review_independence='same-family',acceptance_status='provisional',max_errors=errors,compared_numeric_values=counts,metric_rows=len(frame),score_rows=len(ss),paired_rows=len(boot),gate_rows=len(gates),selected=selected,passing_2014_validation=len(dev),passing_evaluation=sum(x['eligible'] and x['joint_pass'] and x['stage']=='evaluation' for x in gate_rows),calendar_days={str(y):len(d) for y,d in calendars.items()},omitted_input_dates_by_farm=nwp_gaps,all_macro_farms=sorted(macro.farms.unique().tolist()),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'p1_inference_verify.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
