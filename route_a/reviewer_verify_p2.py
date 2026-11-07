"""Fresh reviewer checks: independent formulas, no optimizer or experiment writes."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/route_a/20261107_p2_v040'
P0=ROOT/'experiments/route_a/20261107_p0_v020'
OUT=RUN/'review_trace';OUT.mkdir(exist_ok=True)
Q=np.array([.025,.05,.1,.5,.9,.95,.975])
M=('coverage','width','IS','lower_miss','upper_miss')
S=('WIS','pinball7','median_MAE')
C=('adjacent_crossings','nested_80_90','nested_90_95')
result={'reviewer':'fresh same-family provisional','status':'running','checks':{}}

def compare(name,actual,keys,columns):
    expected=pd.read_csv(RUN/name,dtype={'group':str})
    a=actual.set_index(keys).sort_index();b=expected.set_index(keys).sort_index()
    assert a.index.is_unique and b.index.is_unique and a.index.equals(b.index),name
    av=a[columns].to_numpy(float);bv=b[columns].to_numpy(float)
    np.testing.assert_allclose(av,bv,rtol=0,atol=2e-12,equal_nan=True,err_msg=name)
    err=float(np.nanmax(abs(av-bv)))
    result['checks'][name]={'rows':len(a),'max_absolute_error':err}

def array_metrics(y,lo,hi,alpha):
    low=y<lo;high=y>hi
    return dict(coverage=(~(low|high)).astype(float),width=hi-lo,
        IS=(hi-lo)+(2/alpha)*(np.where(low,lo-y,0)+np.where(high,y-hi,0)),
        lower_miss=low.astype(float),upper_miss=high.astype(float))

def masks(d,ids):
    y=d['y'][ids];change=np.concatenate([y[:,:1]-d['history'][ids,-1:],np.diff(y,axis=1)],axis=1)
    a=[('regime','all',np.ones(y.shape,bool)),('regime','up_ramp',change>.1),
       ('regime','down_ramp',change<-.1),('regime','near_zero',y<=.01)]
    for lo,hi,label in [(0,.01,'0_to_.01'),(.01,.2,'.01_to_.2'),(.2,.8,'.2_to_.8'),(.8,1,'.8_to_1')]:
        a.append(('power',label,((y>=lo) if lo==0 else (y>lo))&(y<=hi)))
    for lead in range(24):a.append(('lead',str(lead+1),np.broadcast_to(np.arange(24)==lead,y.shape)))
    ts=d['origins'][ids,None]+np.arange(1,25)*np.timedelta64(1,'h')
    month=ts.astype('datetime64[M]').astype(str)
    for label in np.unique(month):a.append(('month',label,month==label))
    month_num=ts.astype('datetime64[M]').astype(int)%12+1
    for label,months in [('DJF',[12,1,2]),('MAM',[3,4,5]),('JJA',[6,7,8]),('SON',[9,10,11])]:
        a.append(('season',label,np.isin(month_num,months)))
    return a

def aggregate(frame,keys,columns,counts=()):
    gr=frame.groupby(keys+['seed']);by=gr[list(columns)].mean()
    by['farms']=gr.farm.nunique();by['cells']=gr.cells.sum()
    for c in counts:by[c]=gr[c].sum()
    by=by.reset_index();gr=by.groupby(keys);macro=gr[list(columns)+list(counts)].mean()
    macro['seeds']=gr.seed.nunique();macro['farms']=gr.farms.min();macro['cells_per_seed']=gr.cells.mean()
    for c in columns:macro[c+'_seed_std']=gr[c].std(ddof=1)
    return by,macro.reset_index()

rows=[];scores=[];boundary=[];daily={};rank_rows=[];raw_counts=[];ground=[]
manifest=json.loads((P0/'input_manifest.json').read_text())
source=[Path(x['path']) for x in manifest if x['kind']=='raw_data']
raw14=pd.concat([pd.read_csv(p)[['ZONEID','TIMESTAMP','TARGETVAR']] for p in source if p.name.startswith('Task15_')]+[pd.read_csv(p)[['ZONEID','TIMESTAMP','TARGETVAR']] for p in source if p.name=='solution15_W.csv'],ignore_index=True)
raw14['time']=pd.to_datetime(raw14.TIMESTAMP,format='%Y%m%d %H:%M')
raw12=pd.read_csv(next(p for p in source if p.name=='windpowermeasurements.csv'))
raw12.index=pd.to_datetime(raw12.date.astype(str),format='%Y%m%d%H')
source_hashes=[]
for item in manifest:
    if item['kind']!='raw_data':continue
    p=Path(item['path']);digest=hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest==item['sha256'];source_hashes.append({'path':str(p),'sha256':digest})
result['checks']['raw_source_hashes']={'count':len(source_hashes),'all_equal_to_P0':True}
for year,nfarms in [(2014,10),(2012,7)]:
    for farm in range(1,nfarms+1):
        prior=P0/f'gefcom{year}/farm_{farm:02d}';folder=RUN/f'gefcom{year}/farm_{farm:02d}'
        with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in z.files}
        meta=pd.read_csv(prior/'samples.csv');fc=np.flatnonzero(meta.phase.ne('train'))
        power=raw14.loc[raw14.ZONEID.eq(farm)].set_index('time').TARGETVAR if year==2014 else raw12[f'wp{farm}']
        ts=d['origins'][:,None]+np.arange(1,25)*np.timedelta64(1,'h')
        direct=power.reindex(pd.DatetimeIndex(ts.ravel())).to_numpy().reshape(ts.shape)
        np.testing.assert_equal(direct,d['y'])
        ht=d['origins'][:,None]+np.arange(-23,1)*np.timedelta64(1,'h')
        hist=power.reindex(pd.DatetimeIndex(ht.ravel())).to_numpy().reshape(ht.shape)
        np.testing.assert_equal(hist,d['history'])
        np.testing.assert_array_equal(d['valid'],(np.isfinite(direct)&(direct>=0)&(direct<=1)).all(axis=1))
        ground.append(dict(year=year,farm=farm,issued_days=len(direct),target_cells=int(direct.size),exact=True))
        for family in ['QRF','MLP','TCN','GRU']:
            for seed in ([0] if family=='QRF' else [0,1,2]):
                dest=folder/family/f'seed_{seed}'
                raw=d['pred'] if family=='QRF' else np.full(d['pred'].shape,np.nan)
                if family!='QRF':
                    with np.load(dest/'heads.npz') as z:np.testing.assert_array_equal(z['quantiles'],Q);raw[fc]=z['prediction']
                assert np.isfinite(raw[fc]).all() and (raw[fc]>=0).all() and (raw[fc]<=1).all()
                assert (np.diff(raw[fc],axis=-1)>=0).all()
                raw_counts.append(dict(year=year,farm=farm,model=family,seed=seed,crossings=0))
                for stage in ['validation','evaluation']:
                    ids=np.flatnonzero(meta.phase.eq(stage));y=d['y'][ids];valid=np.broadcast_to(d['valid'][ids,None],y.shape)
                    with np.load(dest/f'{stage}.npz') as z:arr={k:z[k] for k in z.files}
                    np.testing.assert_array_equal(arr['indices'],ids);np.testing.assert_array_equal(arr['pred'][0],raw[ids])
                    np.testing.assert_array_equal(arr['pred'][0,:,:,3],arr['pred'][1,:,:,3])
                    if farm==1 and seed==0:
                        seeds=np.flatnonzero(meta.phase.eq('validation_seed' if stage=='validation' else 'validation'))
                        maxerr=0.;nchecked=0
                        for j,idx in enumerate(ids):
                            pool=np.concatenate((seeds,ids[:j]));now=d['origins'][idx]
                            used=pool[d['valid'][pool]&(d['ends'][pool]<=now)&(d['ends'][pool]>now-np.timedelta64(84,'D'))]
                            for alpha,li,ui in [(.2,2,4),(.1,1,5),(.05,0,6)]:
                                rank=int(np.ceil((len(used)+1)*(1-alpha)-1e-12));assert 1<=rank<=len(used)
                                sc=np.maximum(raw[used,:,li]-d['y'][used],d['y'][used]-raw[used,:,ui])
                                q=np.partition(sc,rank-1,axis=0)[rank-1]
                                lo=raw[idx,:,li]-q;hi=raw[idx,:,ui]+q;bad=lo>hi
                                mid=(raw[idx,:,li]+raw[idx,:,ui])/2
                                lo=np.clip(np.where(bad,mid,lo),0,1);hi=np.clip(np.where(bad,mid,hi),0,1)
                                np.testing.assert_array_equal(lo,arr['pred'][1,j,:,li]);np.testing.assert_array_equal(hi,arr['pred'][1,j,:,ui])
                                prefix=f'{1-alpha:.2f}_';np.testing.assert_array_equal(q,arr[prefix+'correction'][j])
                                assert len(used)==arr[prefix+'n_cal'][j] and d['ends'][used].max()==arr[prefix+'max_label_end'][j]
                                nchecked+=24
                        rank_rows.append(dict(year=year,farm=farm,model=family,seed=seed,stage=stage,lead_ranks=nchecked,max_error=maxerr))
                    for arm,suffix in enumerate(['raw','CQR']):
                        pred=arr['pred'][arm];method=family+'_'+suffix
                        info=dict(year=year,farm=farm,stage=stage,method=method,seed=seed)
                        wis=.5*abs(y-pred[:,:,3]);reg=masks(d,ids)
                        for alpha,li,ui in [(.2,2,4),(.1,1,5),(.05,0,6)]:
                            stat=array_metrics(y,pred[:,:,li],pred[:,:,ui],alpha);wis+=alpha/2*stat['IS']
                            for dim,group,mask in reg:
                                mask=mask&valid
                                if mask.any():rows.append(dict(**info,confidence=1-alpha,dimension=dim,group=group,cells=mask.sum(),days=mask.any(axis=1).sum(),**{k:v[mask].mean() for k,v in stat.items()}))
                        wis/=3.5;error=y[:,:,None]-pred
                        pb=np.where(error>=0,Q*error,(Q-1)*error).mean(axis=-1);mae=abs(y-pred[:,:,3])
                        np.testing.assert_allclose(wis[valid],2*pb[valid],rtol=0,atol=1e-12)
                        s90=array_metrics(y,pred[:,:,1],pred[:,:,5],.1)
                        for dim,group,mask in reg:
                            if dim!='regime':continue
                            mask=mask&valid
                            if not mask.any():continue
                            cross=np.diff(pred,axis=-1)<0
                            scores.append(dict(**info,group=group,cells=mask.sum(),WIS=wis[mask].mean(),pinball7=pb[mask].mean(),median_MAE=mae[mask].mean(),adjacent_crossings=(cross&mask[:,:,None]).sum(),nested_80_90=(((pred[:,:,2]<pred[:,:,1])|(pred[:,:,4]>pred[:,:,5]))&mask).sum(),nested_90_95=(((pred[:,:,1]<pred[:,:,0])|(pred[:,:,5]>pred[:,:,6]))&mask).sum()))
                            if group in ['all','up_ramp','near_zero']:
                                daily[year,farm,stage,method,seed,group]=dict(origins=d['origins'][ids],n=mask.sum(axis=1),**{k:np.where(mask,v,0).sum(axis=1) for k,v in {**s90,'WIS':wis,'median_MAE':mae}.items()})
                        for group,mask in [('all',np.ones(y.shape,bool)),('exact_zero',y==0),('positive_near_zero',(y>0)&(y<=.01)),('near_zero',y<=.01)]:
                            mask=mask&valid
                            if mask.any():boundary.append(dict(**info,group=group,cells=mask.sum(),coverage90=s90['coverage'][mask].mean(),lower_miss90=s90['lower_miss'][mask].mean(),mean_lower90=pred[:,:,1][mask].mean(),lower_zero_fraction90=(pred[:,:,1][mask]==0).mean(),width90=s90['width'][mask].mean(),IS90=s90['IS'][mask].mean()))
        print('VERIFIED_FARM',year,farm,flush=True)
result['checks']['ground_truth']={'farms':ground,'evaluation_type':'real_gt','normalization':'raw dataset capacity-normalized values; no prediction denominators'}
result['checks']['independent_cqr']=rank_rows
result['checks']['raw_noncrossing']={'models':len(raw_counts),'violations':0}
metric=pd.DataFrame(rows);score=pd.DataFrame(scores);bound=pd.DataFrame(boundary)
keys=['year','stage','method','confidence','dimension','group'];sk=['year','stage','method','group']
compare('metrics_per_farm.csv',metric,['year','farm','stage','method','seed','confidence','dimension','group'],['cells','days',*M])
compare('scores_per_farm.csv',score,['year','farm','stage','method','seed','group'],['cells',*S,*C])
compare('boundary_diagnostics.csv',bound,['year','farm','stage','method','seed','group'],[c for c in bound if c not in ['year','farm','stage','method','seed','group']])
ms,mm=aggregate(metric,keys,M);ss,sm=aggregate(score,sk,S,C)
compare('metrics_seed_macro.csv',ms,keys+['seed'],['farms','cells',*M])
compare('metrics_macro.csv',mm,keys,['seeds','farms','cells_per_seed',*M,*[k+'_seed_std' for k in M]])
compare('scores_seed_macro.csv',ss,sk+['seed'],['farms','cells',*S,*C])
compare('scores_macro.csv',sm,sk,['seeds','farms','cells_per_seed',*S,*C,*[k+'_seed_std' for k in S]])
for cur,old in [('QRF_raw','QRF_raw'),('QRF_CQR','QRF_CQR_rolling84')]:
    baseline=pd.read_csv(P0/'metrics_macro.csv',dtype={'group':str});kk=[k for k in keys if k!='method']
    a=mm[mm.method.eq(cur)].set_index(kk).sort_index();b=baseline[baseline.method.eq(old)].set_index(kk).sort_index()
    assert a.index.equals(b.index);np.testing.assert_allclose(a[list(M)],b[list(M)],rtol=0,atol=2e-12)
result['checks']['P0_controls_exact']={'max_absolute_error':float(abs(a[list(M)]-b[list(M)]).to_numpy().max())}

def gates(metrics,proper,per_seed):
    out=[];mt=metrics[metrics.dimension.eq('regime')&metrics.confidence.eq(.9)]
    for year in [2014,2012]:
      for stage in ['validation','evaluation']:
        base=mt[mt.year.eq(year)&mt.stage.eq(stage)&mt.method.eq('QRF_CQR')].set_index('group')
        bs=proper[proper.year.eq(year)&proper.stage.eq(stage)&proper.method.eq('QRF_CQR')].set_index('group')
        for method in ['MLP_CQR','TCN_CQR','GRU_CQR']:
          for seed in ([0,1,2] if per_seed else [-1]):
            a=mt[mt.year.eq(year)&mt.stage.eq(stage)&mt.method.eq(method)];b=proper[proper.year.eq(year)&proper.stage.eq(stage)&proper.method.eq(method)]
            if per_seed:a=a[a.seed.eq(seed)];b=b[b.seed.eq(seed)]
            a=a.set_index('group');b=b.set_index('group');v=a.loc['all'];r=base.loc['all']
            ratios=dict(coverage=v.coverage,width_ratio=v.width/r.width,IS_ratio=v.IS/r.IS,WIS_ratio=b.loc['all','WIS']/bs.loc['all','WIS'],up_coverage_gain=a.loc['up_ramp','coverage']-base.loc['up_ramp','coverage'],zero_coverage_gain=a.loc['near_zero','coverage']-base.loc['near_zero','coverage'],up_IS_ratio=a.loc['up_ramp','IS']/base.loc['up_ramp','IS'],zero_IS_ratio=a.loc['near_zero','IS']/base.loc['near_zero','IS'])
            passed=dict(coverage=v.coverage>=.9,coverage_drop=v.coverage>=r.coverage-.005,width=ratios['width_ratio']<=1.01,IS=ratios['IS_ratio']<=1,WIS=ratios['WIS_ratio']<=1,up_coverage=ratios['up_coverage_gain']>=.01,zero_coverage=ratios['zero_coverage_gain']>=.01,up_IS=ratios['up_IS_ratio']<=1,zero_IS=ratios['zero_IS_ratio']<=1)
            out.append(dict(year=year,stage=stage,method=method,seed=seed,**ratios,joint_pass=all(passed.values()),**{'pass_'+k:v for k,v in passed.items()}))
    frame=pd.DataFrame(out);name='joint_gates_by_seed.csv' if per_seed else 'joint_gates.csv'
    compare(name,frame,['year','stage','method','seed'],[c for c in frame if c not in ['year','stage','method','seed']])
    return frame
g=gates(mm,sm,False);gs=gates(ms,ss,True)
selection=json.loads((RUN/'selection.json').read_text());passing=g[g.year.eq(2014)&g.stage.eq('validation')&g.joint_pass]
selected='QRF_CQR' if passing.empty else passing.sort_values(['WIS_ratio','method']).iloc[0].method
assert selected==selection['selected'] and set(passing.method)==set(selection['eligible_passing'])
result['checks']['selection']={'selected':selected,'mean_gate_pass_count':int(g.joint_pass.sum()),'seed_gate_pass_count':int(gs.joint_pass.sum())}

rng=np.random.default_rng(20261107);boot=[];calendar=[]
comparisons=[('QRF_CQR',f+'_CQR') for f in ['MLP','TCN','GRU']]+[('QRF_raw',f+'_raw') for f in ['MLP','TCN','GRU']]+[('MLP_CQR',f+'_CQR') for f in ['TCN','GRU']]+[('MLP_raw',f+'_raw') for f in ['TCN','GRU']]
for year,nfarms in [(2014,10),(2012,7)]:
    cal=pd.read_csv(P0/f'gefcom{year}/farm_01/calendar.csv');dates=pd.DatetimeIndex(cal.loc[cal.phase.eq('evaluation'),'origin']).to_numpy()
    n=len(dates);assert np.array_equal(dates,pd.date_range(dates.min(),dates.max()).to_numpy())
    starts=rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1));draws=(starts+np.arange(14))%n;draws=draws.reshape(1000,-1)[:,:n]
    # Histogram implementation independently weights day sums/counts, instead of indexing sums per draw.
    weights=np.zeros((1000,n),int)
    for i,draw in enumerate(draws):weights[i]=np.bincount(draw,minlength=n)
    for group in ['all','up_ramp','near_zero']:
        estimates={};obs={}
        for method in ['QRF_raw','QRF_CQR','MLP_raw','MLP_CQR','TCN_raw','TCN_CQR','GRU_raw','GRU_CQR']:
            seeds=[0] if method.startswith('QRF') else [0,1,2]
            for metric_name in [*M,'WIS','median_MAE']:
                vv=[];oo=[]
                for seed in seeds:
                    for farm in range(1,nfarms+1):
                        item=daily[year,farm,'evaluation',method,seed,group];ix=np.searchsorted(dates,item['origins'])
                        sums=np.zeros(n);counts=np.zeros(n);sums[ix]=item[metric_name];counts[ix]=item['n']
                        den=weights@counts;assert (den>0).all()
                        vv.append((weights@sums)/den);oo.append(sums.sum()/counts.sum())
                estimates[method,metric_name]=np.mean(vv,axis=0);obs[method,metric_name]=np.mean(oo)
        for reference,method in comparisons:
            for metric_name in [*M,'WIS','median_MAE']:
                delta=estimates[method,metric_name]-estimates[reference,metric_name];low,high=np.quantile(delta,[.025,.975])
                boot.append(dict(year=year,stage='evaluation',method=method,reference=reference,group=group,metric=metric_name,delta=obs[method,metric_name]-obs[reference,metric_name],ci_low=low,ci_high=high,draws=1000,block_calendar_days=14,neural_seeds_averaged=3))
    calendar.append(dict(year=year,calendar_days=n,full_calendar=True,paired_common_draws=True,absent_issued_dates_zero_count=True))
    print('BOOTSTRAP_VERIFIED',year,flush=True)
compare('paired_calendar_blocks.csv',pd.DataFrame(boot),['year','stage','method','reference','group','metric'],['delta','ci_low','ci_high','draws','block_calendar_days','neural_seeds_averaged'])
result['checks']['bootstrap_calendar']=calendar
result['status']='pass';result['scope']='All 17 farms, 170 model-seed prediction sets, both stages, all declared masks; CQR rank replay both farm01 all four families seed0. No model training rerun.'
(OUT/'independent_numeric_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':{k:v for k,v in result['checks'].items() if isinstance(v,dict) and 'rows' in v},'selection':result['checks']['selection']},indent=2))
