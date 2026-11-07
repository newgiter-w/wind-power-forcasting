"""P2 same-mask raw and signed84-CQR evaluation; no training or model selection."""
from pathlib import Path
import argparse,json,time
import numpy as np
import pandas as pd
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p0_baseline import Q,ALPHAS,groups,metric_arrays,causal_cqr,write,sha
from route_a.p2_train import P0

FAMILIES=('MLP','TCN','GRU')
METHODS=('QRF_raw','QRF_CQR','MLP_raw','MLP_CQR','TCN_raw','TCN_CQR','GRU_raw','GRU_CQR')
METRICS=('coverage','width','IS','lower_miss','upper_miss')


def collect(rows,scores,daily,boundary,year,farm,stage,family,seed,d,ids,heads):
    y=d['y'][ids];valid=np.broadcast_to(d['valid'][ids,None],y.shape);masks=groups(d,ids)
    for arm,label in enumerate(('raw','CQR')):
        method=family+'_'+label;pred=heads[arm];median=pred[:,:,3];wis=.5*abs(y-median)
        for alpha in ALPHAS:
            qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
            stat=metric_arrays(y,pred[:,:,qi],pred[:,:,qj],alpha);wis+=alpha/2*stat['IS']
            for dimension,group,mask in masks:
                m=valid&mask
                if m.any():rows.append(dict(year=year,farm=farm,stage=stage,method=method,seed=seed,confidence=1-alpha,
                    dimension=dimension,group=group,cells=int(m.sum()),days=int(m.any(axis=1).sum()),
                    **{k:float(a[m].mean()) for k,a in stat.items()}))
        wis/=3.5;error=y[:,:,None]-pred
        pinball=np.maximum(Q*error,(Q-1)*error).mean(axis=-1)
        np.testing.assert_allclose(wis[valid],2*pinball[valid],rtol=0,atol=1e-12)
        s90=metric_arrays(y,pred[:,:,1],pred[:,:,5],.1)
        mae=abs(y-median)
        for dimension,group,mask in masks:
            if dimension!='regime':continue
            m=valid&mask
            if not m.any():continue
            scores.append(dict(year=year,farm=farm,stage=stage,method=method,seed=seed,group=group,cells=int(m.sum()),
                WIS=float(wis[m].mean()),pinball7=float(pinball[m].mean()),median_MAE=float(mae[m].mean()),
                adjacent_crossings=int(((np.diff(pred,axis=-1)<0)&m[:,:,None]).sum()),
                nested_80_90=int((((pred[:,:,2]<pred[:,:,1])|(pred[:,:,4]>pred[:,:,5]))&m).sum()),
                nested_90_95=int((((pred[:,:,1]<pred[:,:,0])|(pred[:,:,5]>pred[:,:,6]))&m).sum())))
            if group in ('all','up_ramp','near_zero'):
                daily[year,farm,stage,method,seed,group]=dict(origins=d['origins'][ids],n=m.sum(axis=1),
                    **{k:np.where(m,a,0).sum(axis=1) for k,a in {**s90,'WIS':wis,'median_MAE':mae}.items()})
        for label,mask in [('all',np.ones(y.shape,bool)),('exact_zero',y==0),('positive_near_zero',(y>0)&(y<=.01)),('near_zero',y<=.01)]:
            m=valid&mask
            if m.any():boundary.append(dict(year=year,farm=farm,stage=stage,method=method,seed=seed,group=label,cells=int(m.sum()),
                coverage90=float(s90['coverage'][m].mean()),lower_miss90=float(s90['lower_miss'][m].mean()),
                mean_lower90=float(pred[:,:,1][m].mean()),lower_zero_fraction90=float((pred[:,:,1][m]==0).mean()),
                width90=float(s90['width'][m].mean()),IS90=float(s90['IS'][m].mean())))


def aggregate(frame,keys,columns,seed_name,macro_name,out,count_columns=()):
    # First farm-macro within each fixed seed, then average seeds. Counts are per seed.
    spec={k:(k,'mean') for k in columns};spec.update({k:(k,'sum') for k in count_columns})
    by_seed=frame.groupby(keys+['seed']).agg(farms=('farm','nunique'),cells=('cells','sum'),**spec).reset_index()
    by_seed.to_csv(out/seed_name,index=False)
    spec={k:(k,'mean') for k in columns+tuple(count_columns)}
    spec.update({k+'_seed_std':(k,'std') for k in columns})
    macro=by_seed.groupby(keys).agg(seeds=('seed','nunique'),farms=('farms','min'),cells_per_seed=('cells','mean'),**spec).reset_index()
    macro.to_csv(out/macro_name,index=False)


def bootstrap(daily,out):
    rng=np.random.default_rng(20261107);rows=[]
    comparisons=[('QRF_CQR',m+'_CQR') for m in FAMILIES]+[('QRF_raw',m+'_raw') for m in FAMILIES]+[
        ('MLP_CQR',m+'_CQR') for m in ('TCN','GRU')]+[('MLP_raw',m+'_raw') for m in ('TCN','GRU')]
    for year in (2014,2012):
        farms=sorted({k[1] for k in daily if k[0]==year})
        cal=pd.read_csv(P0/f'gefcom{year}/farm_01/calendar.csv');dates=pd.to_datetime(cal.loc[cal.phase.eq('evaluation'),'origin']).to_numpy()
        n=len(dates);assert n==len(pd.date_range(dates.min(),dates.max()))
        draws=((rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1))+np.arange(14))%n).reshape(1000,-1)[:,:n]
        for group in ('all','up_ramp','near_zero'):
            observed={};estimates={}
            for method in METHODS:
                seeds=(0,) if method.startswith('QRF') else (0,1,2)
                for metric in (*METRICS,'WIS','median_MAE'):
                    obs=[];values=[]
                    for seed in seeds:
                        for farm in farms:
                            a=daily[year,farm,'evaluation',method,seed,group]
                            pos=np.searchsorted(dates,a['origins']);s=np.zeros(n);c=np.zeros(n)
                            s[pos]=a[metric];c[pos]=a['n'];denom=c[draws].sum(axis=1);assert (denom>0).all()
                            obs.append(s.sum()/c.sum());values.append(s[draws].sum(axis=1)/denom)
                    observed[method,metric]=float(np.mean(obs));estimates[method,metric]=np.mean(values,axis=0)
            for reference,method in comparisons:
                for metric in (*METRICS,'WIS','median_MAE'):
                    delta=estimates[method,metric]-estimates[reference,metric];lo,hi=np.quantile(delta,[.025,.975])
                    rows.append(dict(year=year,stage='evaluation',method=method,reference=reference,group=group,metric=metric,
                        delta=observed[method,metric]-observed[reference,metric],ci_low=float(lo),ci_high=float(hi),draws=1000,
                        block_calendar_days=14,neural_seeds_averaged=3))
    pd.DataFrame(rows).to_csv(out/'paired_calendar_blocks.csv',index=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();out=require_local_write(ROOT/args.run)
    assert json.loads((out/'training_completion.json').read_text(encoding='utf-8'))['status']=='complete'
    assert not (out/'completion.json').exists()
    start=time.perf_counter();rows=[];scores=[];daily={};boundary=[];checks=[]
    write(out/'evaluation_code_manifest.json',[dict(path=f,sha256=sha(ROOT/f)) for f in ('route_a/p2_evaluate.py','route_a/p0_baseline.py')])
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            prior=P0/f'gefcom{year}/farm_{farm:02d}';folder=out/f'gefcom{year}/farm_{farm:02d}'
            with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in ('y','history','valid','origins','ends')};qrf=z['pred']
            meta=pd.read_csv(prior/'samples.csv');forecast=np.flatnonzero(~meta.phase.eq('train'))
            models=[('QRF',0)]+[(m,seed) for m in FAMILIES for seed in (0,1,2)]
            for family,seed in models:
                dest=folder/family/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                if family=='QRF':raw=qrf
                else:
                    raw=np.full(qrf.shape,np.nan)
                    with np.load(dest/'heads.npz') as z:
                        np.testing.assert_array_equal(z['quantiles'],Q);raw[forecast]=z['prediction']
                    assert np.isfinite(raw[forecast]).all() and (np.diff(raw[forecast],axis=-1)>=0).all()
                witness=dict(year=year,farm=farm,model=family,seed=seed,raw_nontrain_crossings=int((np.diff(raw[forecast],axis=-1)<0).sum()))
                for stage in ('validation','evaluation'):
                    ids=np.flatnonzero(meta.phase.eq(stage));seed_ids=np.flatnonzero(meta.phase.eq('validation_seed' if stage=='validation' else 'validation'))
                    heads=np.stack([raw[ids].copy(),raw[ids].copy()]);trace={}
                    with np.load(prior/f'{stage}_calibration.npz') as original:
                        np.testing.assert_array_equal(original['indices'],ids)
                        for alpha in ALPHAS:
                            confidence=1-alpha;prefix=f'QRF_CQR_rolling84_{confidence:.2f}'
                            result=causal_cqr(d,raw,seed_ids,ids,alpha,True)
                            np.testing.assert_array_equal(result['n_cal'],original[prefix+'_n_cal'])
                            np.testing.assert_array_equal(result['max_label_end'],original[prefix+'_max_label_end'])
                            qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
                            heads[1,:,:,qi]=result['lo'];heads[1,:,:,qj]=result['hi']
                            for key in ('n_cal','max_label_end','correction'):trace[f'{confidence:.2f}_{key}']=result[key]
                            if family=='QRF':
                                np.testing.assert_array_equal(result['lo'],original[prefix+'_lo'])
                                np.testing.assert_array_equal(result['hi'],original[prefix+'_hi'])
                        np.savez_compressed(dest/f'{stage}.npz',indices=ids,methods=np.array([family+'_raw',family+'_CQR']),
                            quantiles=Q,pred=heads,**trace)
                    collect(rows,scores,daily,boundary,year,farm,stage,family,seed,d,ids,heads)
                checks.append(witness)
            print('P2_SCORED',year,farm,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(out/'metrics_per_farm.csv',index=False)
    aggregate(frame,['year','stage','method','confidence','dimension','group'],METRICS,
        'metrics_seed_macro.csv','metrics_macro.csv',out)
    ss=pd.DataFrame(scores);ss.to_csv(out/'scores_per_farm.csv',index=False)
    aggregate(ss,['year','stage','method','group'],('WIS','pinball7','median_MAE'),
        'scores_seed_macro.csv','scores_macro.csv',out,('adjacent_crossings','nested_80_90','nested_90_95'))
    pd.DataFrame(boundary).to_csv(out/'boundary_diagnostics.csv',index=False)
    write(out/'calibration_checks.json',checks)
    bootstrap(daily,out)
    keys=list(daily);write(out/'daily_keys.json',[list(k) for k in keys])
    np.savez_compressed(out/'daily_sufficient.npz',**{f'{i}_{name}':a for i,k in enumerate(keys) for name,a in daily[k].items()})
    write(out/'completion.json',dict(status='complete',version='0.4.0',farms=17,architectures=list(FAMILIES),neural_seeds=[0,1,2],
        neural_final_models=153,QRF_refits=0,methods=list(METHODS),evaluation_seconds=time.perf_counter()-start,
        scope='historical seed-mean comparisons, no ensemble; all methods share P0 eligible/scored dates and signed84 CQR'))
    print('P2_EVALUATION_COMPLETE',out,round(time.perf_counter()-start,2),flush=True)


if __name__=='__main__':main()
