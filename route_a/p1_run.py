"""Frozen QRF recalibration experiment. All outputs are new Route-A artifacts."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='2'
from pathlib import Path
import argparse,gc,json,time
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import save_npz
from threadpoolctl import threadpool_limits
from online_calibration import simulate_independent
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write,new_run_dir
from route_a.p0_baseline import Q,ALPHAS,groups,metric_arrays,sha,write
from route_a.p1_distribution import FrozenECDF,inverse,distribution_stats,arrived,local_pool,mixed_values,tail_quantile

ARMS=('QRF_linear_raw','CQR_signed84','DualTail84','QRF_ECDF_raw','PIT84','PIT84_state','DualTail84_state')
REFERENCE='CQR_signed84'
P0=ROOT/'experiments/route_a/20261107_p0_v020'
MODELS=ROOT/'out_gefcom_revision_20261104/qrf'
INPUT=ROOT/'out_gefcom_pooling_20261105/inputs'


def states_from_predictions(raw,p_zero,history):
    rise=np.diff(raw[...,3],axis=1,prepend=history[:,-1,None])
    return np.where(p_zero>=.5,0,np.where(rise>.1,1,2)).astype('int8')


def stage_predictions(d,raw,ecdf,weights,support,pit,states,forecast,ids,seed,baseline):
    """Only arrived pools influence forecasts; truth additionally enters scoring."""
    n,h=len(ids),24
    output=np.full((len(ARMS),n,h,len(Q)),np.nan)
    crps=np.full((len(ARMS),n,h),np.nan)
    output[0]=raw[ids];output[3]=ecdf[ids]
    output[1]=raw[ids]
    for ai,alpha in enumerate(ALPHAS):
        qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
        output[1,:,:,qi]=baseline[f'CQR_lo_{1-alpha:.2f}']
        output[1,:,:,qj]=baseline[f'CQR_hi_{1-alpha:.2f}']
    offset=np.full(len(raw),-1,dtype=int);offset[forecast]=np.arange(len(forecast))
    ncal=[];max_end=[];local_days=np.empty((n,h),int);lambdas=np.empty((n,h))
    collapse=np.zeros((len(ARMS),len(ALPHAS)),int)
    for position,idx in enumerate(ids):
        used=arrived(d,seed,ids,position);ncal.append(len(used));max_end.append(d['ends'][used].max())
        for lead in range(h):
            row=offset[idx]*24+lead
            a,b=weights.indptr[row:row+2]
            xx=support[weights.indices[a:b]];ww=weights.data[a:b]
            local_i,local_h,weight,ndays=local_pool(used,lead,states[idx,lead],states)
            local_days[position,lead]=ndays;lambdas[position,lead]=weight
            for arm,shrink in ((4,False),(5,True)):
                values,prob=mixed_values(pit[used,lead],pit[local_i,local_h],weight if shrink else 0.)
                samples=inverse(xx,ww,values,boundary=True)
                truth=d['y'][idx,lead]
                q,c=distribution_stats(samples,prob,float(truth) if np.isfinite(truth) else 0.,Q)
                output[arm,position,lead]=q
                crps[arm,position,lead]=c if np.isfinite(truth) else np.nan
            for arm,shrink in ((2,False),(6,True)):
                output[arm,position,lead,3]=raw[idx,lead,3]
                for ai,alpha in enumerate(ALPHAS):
                    qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
                    sl=raw[used,lead,qi]-d['y'][used,lead]
                    su=d['y'][used,lead]-raw[used,lead,qj]
                    rank=int(np.ceil((len(used)+1)*(1-alpha/2)-1e-12))
                    assert 1<=rank<=len(used)
                    if shrink and weight>0:
                        lv,lp=mixed_values(sl,raw[local_i,local_h,qi]-d['y'][local_i,local_h],weight)
                        uv,up=mixed_values(su,d['y'][local_i,local_h]-raw[local_i,local_h,qj],weight)
                        ql=tail_quantile(lv,lp,rank/len(used));qu=tail_quantile(uv,up,rank/len(used))
                    else:
                        ql=np.sort(sl)[rank-1];qu=np.sort(su)[rank-1]
                    lo=raw[idx,lead,qi]-ql;hi=raw[idx,lead,qj]+qu
                    if lo>hi:
                        lo=hi=(raw[idx,lead,qi]+raw[idx,lead,qj])/2
                        collapse[arm,ai]+=1
                    output[arm,position,lead,qi]=np.clip(lo,0,1)
                    output[arm,position,lead,qj]=np.clip(hi,0,1)
    assert np.isfinite(output).all() and (output>=0).all() and (output<=1).all()
    for arm in (3,4,5):assert (np.diff(output[arm],axis=-1)>=-1e-12).all()
    return output,crps,dict(n_cal=np.array(ncal),max_label_end=np.array(max_end),
        local_days=local_days,lambdas=lambdas,midpoint_collapses=collapse)


def score(d,ids,pred,crps,year,farm,stage,rows,scores,daily):
    y=d['y'][ids];valid=np.broadcast_to(d['valid'][ids,None],y.shape)
    allgroups=groups(d,ids)
    for arm,name in enumerate(ARMS):
        median=pred[arm,:,:,3];raw_median=pred[0,:,:,3]
        wis=.5*abs(y-median);wis_common=.5*abs(y-raw_median)
        for alpha in ALPHAS:
            qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
            lo,hi=pred[arm,:,:,qi],pred[arm,:,:,qj]
            stats=metric_arrays(y,lo,hi,alpha);wis+=alpha/2*stats['IS'];wis_common+=alpha/2*stats['IS']
            for dimension,group,mask in allgroups:
                m=mask&valid
                if m.any():
                    rows.append(dict(year=year,farm=farm,stage=stage,method=name,confidence=1-alpha,dimension=dimension,group=group,
                        cells=int(m.sum()),days=int(m.any(axis=1).sum()),**{k:float(v[m].mean()) for k,v in stats.items()}))
        wis/=3.5;wis_common/=3.5
        error=y[:,:,None]-pred[arm]
        pinball=np.maximum(Q*error,(Q-1)*error).mean(axis=-1)
        np.testing.assert_allclose(wis[valid],2*pinball[valid],rtol=0,atol=1e-12)
        stat90=metric_arrays(y,pred[arm,:,:,1],pred[arm,:,:,5],.1)
        for dimension,group,mask in allgroups:
            if dimension!='regime':continue
            m=mask&valid
            if not m.any():continue
            rec=dict(year=year,farm=farm,stage=stage,method=name,group=group,cells=int(m.sum()),days=int(m.any(axis=1).sum()),
                WIS=float(wis[m].mean()),common_median_WIS=float(wis_common[m].mean()),pinball7=float(pinball[m].mean()),
                median_MAE=float(abs(y-median)[m].mean()),
                CRPS=float(crps[arm][m].mean()) if arm in (3,4,5) else None,
                nested_80_90=int((((pred[arm,:,:,2]<pred[arm,:,:,1])|(pred[arm,:,:,4]>pred[arm,:,:,5]))&m).sum()),
                nested_90_95=int((((pred[arm,:,:,1]<pred[arm,:,:,0])|(pred[arm,:,:,5]>pred[arm,:,:,6]))&m).sum()))
            scores.append(rec)
            if group in ('all','up_ramp','near_zero'):
                stats={**stat90,'WIS':wis,'common_median_WIS':wis_common}
                if arm in (3,4,5):stats['CRPS']=crps[arm]
                daily[year,farm,stage,name,group]=dict(origins=d['origins'][ids],n=m.sum(axis=1),
                    **{k:np.where(m,v,0).sum(axis=1) for k,v in stats.items()})


def paired_blocks(daily,out):
    rng=np.random.default_rng(20261107);rows=[]
    for year in (2014,2012):
        farms=sorted({k[1] for k in daily if k[0]==year})
        calendar=pd.read_csv(P0/f'gefcom{year}/farm_01/calendar.csv')
        dates=pd.to_datetime(calendar.loc[calendar.phase.eq('evaluation'),'origin']).to_numpy()
        n=len(dates);assert n==len(pd.date_range(dates.min(),dates.max()))
        draw=((rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1))+np.arange(14))%n).reshape(1000,-1)[:,:n]
        for group in ('all','up_ramp','near_zero'):
            estimates={};observed={}
            for method in ARMS:
                for metric in ('coverage','width','IS','WIS','CRPS'):
                    if metric=='CRPS' and method not in ARMS[3:6]:continue
                    values=[];obs=[]
                    for farm in farms:
                        d=daily[year,farm,'evaluation',method,group]
                        pos=np.searchsorted(dates,d['origins']);s=np.zeros(n);c=np.zeros(n)
                        s[pos]=d[metric];c[pos]=d['n'];denom=c[draw].sum(axis=1)
                        assert (denom>0).all()
                        values.append(s[draw].sum(axis=1)/denom);obs.append(s.sum()/c.sum())
                    estimates[method,metric]=np.mean(values,axis=0);observed[method,metric]=float(np.mean(obs))
            comparisons=[(REFERENCE,m) for m in ARMS if m!=REFERENCE]+[('QRF_ECDF_raw',m) for m in ('PIT84','PIT84_state')]
            for reference,method in comparisons:
                for metric in ('coverage','width','IS','WIS','CRPS'):
                    if (reference,metric) not in estimates or (method,metric) not in estimates:continue
                    diff=estimates[method,metric]-estimates[reference,metric];lo,hi=np.quantile(diff,[.025,.975])
                    rows.append(dict(year=year,stage='evaluation',group=group,method=method,reference=reference,metric=metric,
                        delta=observed[method,metric]-observed[reference,metric],ci_low=float(lo),ci_high=float(hi),
                        block_calendar_days=14,draws=1000))
    pd.DataFrame(rows).to_csv(out/'paired_calendar_blocks.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');args=p.parse_args()
    out=require_local_write(ROOT/args.out) if args.out else new_run_dir('p1_fixed_qrf')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'completion.json').exists()
    protocol=json.loads((ROOT/'route_a/p1_protocol.json').read_text(encoding='utf-8'));write(out/'protocol.json',protocol)
    sources=['route_a/p1_run.py','route_a/p1_distribution.py','route_a/p1_protocol.json','route_a/p0_baseline.py','online_calibration.py']
    write(out/'code_manifest.json',[dict(path=f,sha256=sha(ROOT/f)) for f in sources])
    print('OUTPUT',out,flush=True);started=time.perf_counter();rows=[];scores=[];daily={};checks=[];inputs=[];state_rows=[]
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            t0=time.perf_counter();folder=out/f'gefcom{year}/farm_{farm:02d}';folder.mkdir(parents=True,exist_ok=True)
            prior=P0/f'gefcom{year}/farm_{farm:02d}';old=MODELS/f'gefcom{year}/farm_{farm:02d}/leaf_10/seed_0'
            cached=INPUT/f'gefcom{year}/farm_{farm:02d}'
            with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in z.files}
            raw=d.pop('pred');meta=pd.read_csv(prior/'samples.csv');d['meta']=meta
            train=np.flatnonzero(meta.phase.eq('train')&d['valid']);forecast=np.flatnonzero(~meta.phase.eq('train'))
            assert d['ends'][train].max()<=d['origins'][forecast].min()
            with np.load(cached/'features.npz') as z:dense=z['QRF']
            model_hash=sha(old/'model.joblib');model=joblib.load(old/'model.joblib')['model'];model.n_jobs=2
            assert model.n_estimators==500 and model.min_samples_leaf==10 and model.random_state==0
            # Fixed native-qrf witness; do not replace the original linear representation.
            witness_days=forecast[np.linspace(0,len(forecast)-1,8,dtype=int)]
            cells=(witness_days[:,None]*24+np.arange(24)).ravel()
            witness=model.predict(dense[cells],quantiles=[.05,.95],weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True)
            np.testing.assert_allclose(witness,raw[witness_days][:,:,[1,5]].reshape(-1,2),rtol=0,atol=1e-12)
            frozen=FrozenECDF(model,dense[(train[:,None]*24+np.arange(24)).ravel()],d['y'][train].ravel())
            flat=(forecast[:,None]*24+np.arange(24)).ravel()
            weights=frozen.weights(dense[flat]);support=frozen.support
            print('WEIGHTS_READY',year,farm,'nnz',weights.nnz,flush=True)
            save_npz(folder/'forecast_weights.npz',weights)
            pit=np.full(d['y'].shape,np.nan);pzero=np.full(d['y'].shape,np.nan)
            ecdf=np.full(raw.shape,np.nan);raw_crps=np.full(d['y'].shape,np.nan)
            u=np.random.default_rng(20261107+year*100+farm).random(d['y'].shape)
            for pos,day in enumerate(forecast):
                for lead in range(24):
                    row=pos*24+lead;a,b=weights.indptr[row:row+2]
                    xx=support[weights.indices[a:b]];ww=weights.data[a:b];yy=d['y'][day,lead]
                    ecdf[day,lead],c=distribution_stats(xx,ww,float(yy) if np.isfinite(yy) else 0.,Q)
                    raw_crps[day,lead]=c if np.isfinite(yy) else np.nan
                    pzero[day,lead]=ww[xx<=.01].sum()/ww.sum()
                    if np.isfinite(yy):
                        below=ww[xx<yy].sum()/ww.sum();at=ww[xx==yy].sum()/ww.sum()
                        pit[day,lead]=np.clip(below+u[day,lead]*at,0,1)
            states=states_from_predictions(raw,pzero,d['history'])
            np.savez_compressed(folder/'distribution_inputs.npz',support=support,forecast=forecast,pit=pit,p_zero=pzero,
                states=states,uniform=u,raw_ecdf_quantiles=ecdf,raw_ecdf_crps=raw_crps,origins=d['origins'])
            witness_info=dict(year=year,farm=farm,forest_sha256_before=model_hash,forest_not_refit=True,
                native_quantile_max_error=float(abs(witness-raw[witness_days][:,:,[1,5]].reshape(-1,2)).max()),
                weights_nnz=int(weights.nnz),support_values=len(support),weights_sum_max_error=float(abs(np.asarray(weights.sum(axis=1)).ravel()-1).max()))
            for stage in ('validation','evaluation'):
                ids=np.flatnonzero(meta.phase.eq(stage));seed=np.flatnonzero(meta.phase.eq('validation_seed' if stage=='validation' else 'validation'))
                base={}
                with np.load(prior/f'{stage}_calibration.npz') as z:
                    np.testing.assert_array_equal(z['indices'],ids)
                    for alpha in ALPHAS:
                        for end in ('lo','hi'):base[f'CQR_{end}_{1-alpha:.2f}']=z[f'QRF_CQR_rolling84_{1-alpha:.2f}_{end}']
                pred,c,trace=stage_predictions(d,raw,ecdf,weights,support,pit,states,forecast,ids,seed,base)
                c[3]=raw_crps[ids]
                legacy=simulate_independent(d['origins'],d['ends'],d['y'],raw[...,1],raw[...,5],d['valid'],seed,ids,'rolling',84)
                err=max(np.max(abs(pred[2,:,:,1]-np.clip(legacy['lo'],0,1))),np.max(abs(pred[2,:,:,5]-np.clip(legacy['hi'],0,1))))
                assert err<1e-12,err
                witness_info[stage+'_dual90_legacy_max_error']=float(err)
                np.savez_compressed(folder/f'{stage}.npz',methods=np.array(ARMS),quantiles=Q,indices=ids,pred=pred,crps=c,**trace)
                score(d,ids,pred,c,year,farm,stage,rows,scores,daily)
                good=np.broadcast_to(d['valid'][ids,None],states[ids].shape)
                for state,label in ((0,'predicted_near_zero'),(1,'predicted_up_ramp'),(2,'other')):
                    m=(states[ids]==state)&good
                    state_rows.append(dict(year=year,farm=farm,stage=stage,state=label,cells=int(m.sum()),
                        all_cells=int(good.sum()),local_fallback_fraction=float((trace['lambdas'][m]==0).mean()) if m.any() else None,
                        mean_lambda=float(trace['lambdas'][m].mean()) if m.any() else None,
                        mean_local_distinct_days=float(trace['local_days'][m].mean()) if m.any() else None))
            witness_info['forest_sha256_after']=sha(old/'model.joblib');assert witness_info['forest_sha256_after']==model_hash
            checks.append(witness_info);write(folder/'checks.json',witness_info)
            for path in (old/'model.joblib',cached/'features.npz',prior/'predictions.npz',prior/'samples.csv',prior/'validation_calibration.npz',prior/'evaluation_calibration.npz'):
                inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size))
            print('FARM_DONE',year,farm,'seconds',round(time.perf_counter()-t0,2),flush=True)
            del model,frozen,weights,dense,raw,d,ecdf,pit;gc.collect()
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_per_farm.csv',index=False)
    keys=['year','stage','method','confidence','dimension','group']
    macro=frame.groupby(keys).agg(farms=('farm','nunique'),cells=('cells','sum'),**{k:(k,'mean') for k in ('coverage','width','IS','lower_miss','upper_miss')}).reset_index()
    macro.to_csv(out/'metrics_macro.csv',index=False)
    ss=pd.DataFrame(scores);ss.to_csv(out/'scores_per_farm.csv',index=False)
    ss.groupby(['year','stage','method','group']).agg(farms=('farm','nunique'),cells=('cells','sum'),
        **{k:(k,'mean') for k in ('WIS','common_median_WIS','pinball7','median_MAE','CRPS')},
        nested_80_90=('nested_80_90','sum'),nested_90_95=('nested_90_95','sum')).reset_index().to_csv(out/'scores_macro.csv',index=False)
    pd.DataFrame(state_rows).to_csv(out/'state_support.csv',index=False)
    write(out/'forest_checks.json',checks);write(out/'input_manifest.json',inputs)
    paired_blocks(daily,out)
    dailykeys=list(daily);write(out/'daily_keys.json',[list(k) for k in dailykeys])
    np.savez_compressed(out/'daily_sufficient.npz',**{f'{i}_{name}':a for i,k in enumerate(dailykeys) for name,a in daily[k].items()})
    write(out/'completion.json',dict(status='complete',version='0.3.0',farms=17,methods=list(ARMS),new_forests_fit=0,
        forest_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-started,scope='retrospective, fixed seed0 forest and one fixed randomized-PIT realization'))
    print('COMPLETE',out,'seconds',round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
