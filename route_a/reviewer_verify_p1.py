"""Independent P1 artifact audit: no study rerun, no training, no production helpers."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
import hashlib,json,gc
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import load_npz
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/route_a/20261107_p1_v030'
P0=ROOT/'experiments/route_a/20261107_p0_v020'
OUT=RUN/'review_trace'
Q=np.array([.025,.05,.1,.5,.9,.95,.975])
R={'method':'independent formulas; production P1 helpers not imported','max_errors':{},'counts':{}}

def zload(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}

def same(name,a,b,tol=2e-12):
    a,b=np.asarray(a),np.asarray(b)
    np.testing.assert_allclose(a,b,rtol=0,atol=tol,equal_nan=True,err_msg=name)
    delta=np.abs(a-b) if np.issubdtype(a.dtype,np.number) else np.array([0.])
    err=float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.
    R['max_errors'][name]=max(R['max_errors'].get(name,0.),err)

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def quant(x,w,p,bound=False):
    # Stable inverse empirical CDF, with only machine-roundoff tolerance at ranks.
    order=np.argsort(x,kind='stable');x=np.asarray(x)[order];w=np.asarray(w)[order]
    cum=np.cumsum(w)/np.sum(w)
    idx=np.searchsorted(cum,np.asarray(p)-1e-14,side='left').clip(0,len(x)-1)
    v=x[idx]
    if bound:v=np.where(np.asarray(p)<=0,0,np.where(np.asarray(p)>=1,1,v))
    return v

def crps_pair(x,w,y):
    w=np.asarray(w)/np.sum(w);x=np.asarray(x)
    return np.dot(w,abs(x-y))-.5*np.sum(w[:,None]*w[None,:]*abs(x[:,None]-x[None,:]))

def cal_ids(d,meta,ids,pos,stage):
    seed=np.flatnonzero(meta.phase.eq('validation_seed' if stage=='validation' else 'validation'))
    cand=np.concatenate((seed,ids[:pos]));now=d['origins'][ids[pos]]
    return cand[(d['ends'][cand]<=now)&(d['ends'][cand]>now-np.timedelta64(84,'D'))&d['valid'][cand]]

def pool(used,lead,state,states):
    cells=[(int(i),j) for i in used for j in range(lead//6*6,lead//6*6+6) if states[i,j]==state]
    nd=len({i for i,j in cells});lam=min(.8,nd/(nd+20)) if nd>=20 and len(cells)>=40 else 0.
    return np.array([i for i,j in cells],int),np.array([j for i,j in cells],int),nd,lam

def mix(g,l,lam):
    if lam==0:return g,np.ones(len(g))/len(g)
    return np.r_[g,l],np.r_[np.full(len(g),(1-lam)/len(g)),np.full(len(l),lam/len(l))]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifests={}
    for name in ('input_manifest','code_manifest'):
        entries=json.loads((RUN/f'{name}.json').read_text())
        for e in entries:assert digest(ROOT/e['path'])==e['sha256'],e['path']
        manifests[name]=len(entries)
    old={str(Path(e['path'])):e['sha256'] for e in json.loads((P0/'input_manifest.json').read_text())}
    forests=json.loads((RUN/'forest_checks.json').read_text())
    for f in forests:
        p=Path(f"out_gefcom_revision_20261104/qrf/gefcom{f['year']}/farm_{f['farm']:02d}/leaf_10/seed_0/model.joblib")
        assert old[str(p)]==f['forest_sha256_before']==f['forest_sha256_after']==digest(ROOT/p)
    R['manifests_verified']=manifests;R['forest_hash_continuity']=len(forests)
    R['analysis_source_hash_verified']=digest(ROOT/'route_a/p1_analyze.py')==json.loads((RUN/'analysis_completion.json').read_text())['source_sha256']
    assert R['analysis_source_hash_verified']
    assert json.loads((RUN/'protocol.json').read_text())==json.loads((ROOT/'route_a/p1_protocol.json').read_text())
    R['unit_command']={'command':'& .venv/Scripts/python.exe tools/run_route_a.py --script route_a/check_p1.py','exit_code':0,'doc_reality_divergence':None,'scope':'synthetic unit witness only, executed verbatim by reviewer'}
    total_base=total_days=sample_cells=forest_cells=pit_cells=gt_cells=0
    details=[]
    for year,nfarms in ((2014,10),(2012,7)):
      for farm in range(1,nfarms+1):
        f=RUN/f'gefcom{year}/farm_{farm:02d}';p=P0/f'gefcom{year}/farm_{farm:02d}'
        d=zload(p/'predictions.npz');meta=pd.read_csv(p/'samples.csv');dist=zload(f/'distribution_inputs.npz')
        states=np.where(dist['p_zero']>=.5,0,np.where(np.diff(d['pred'][...,3],axis=1,prepend=d['history'][:,-1,None])>.1,1,2))
        np.testing.assert_array_equal(states,dist['states'])
        same('uniforms',np.random.default_rng(20261107+year*100+farm).random(d['y'].shape),dist['uniform'])
        forecast=np.flatnonzero(~meta.phase.eq('train'));np.testing.assert_array_equal(forecast,dist['forecast'])
        for stage in ('validation','evaluation'):
            a=zload(f/f'{stage}.npz');base=zload(p/f'{stage}_calibration.npz');ids=np.flatnonzero(meta.phase.eq(stage))
            np.testing.assert_array_equal(ids,a['indices']);np.testing.assert_array_equal(ids,base['indices'])
            same('P0_raw_predictions',a['pred'][0],d['pred'][ids]);same('CQR_median',a['pred'][1,:,:,3],d['pred'][ids,:,3])
            for alpha,li,ui in ((.2,2,4),(.1,1,5),(.05,0,6)):
                same('P0_CQR_endpoints',a['pred'][1,:,:,li],base[f'QRF_CQR_rolling84_{1-alpha:.2f}_lo'])
                same('P0_CQR_endpoints',a['pred'][1,:,:,ui],base[f'QRF_CQR_rolling84_{1-alpha:.2f}_hi'])
            assert np.all(np.diff(a['pred'][[3,4,5]],axis=-1)>=-1e-12)
            total_base+=a['pred'][0].size+a['pred'][1].size
            for pos,idx in enumerate(ids):
                used=cal_ids(d,meta,ids,pos,stage)
                assert idx not in used and len(used)==a['n_cal'][pos]
                assert max(d['ends'][used])==a['max_label_end'][pos]<=d['origins'][idx]
                assert min(np.ceil((len(used)+1)*(1-np.array([.2,.1,.05])/2)-1e-12))>=1
                assert max(np.ceil((len(used)+1)*(1-np.array([.2,.1,.05])/2)-1e-12))<=len(used)
                total_days+=1
        if farm!=1:continue
        # Dataset CSV ground truth, independent of saved model predictions.
        if year==2014:
            rawroot=Path('D:/Study/wind_datasets/nwp_benchmarks/gefcom2014_wind/data/Wind')
            paths=list((rawroot/'Task 15').rglob('Task15_W_Zone1.csv'))
            train=pd.concat([pd.read_csv(v) for v in paths])
            sol=pd.read_csv(rawroot/'Solution to Task 15/solution15_W.csv');sol=sol[sol.ZONEID.eq(1)]
            raw=pd.concat([train[['TIMESTAMP','TARGETVAR']],sol[['TIMESTAMP','TARGETVAR']]])
            power=pd.Series(raw.TARGETVAR.to_numpy(),index=pd.to_datetime(raw.TIMESTAMP,format='%Y%m%d %H:%M'))
        else:
            raw=pd.read_csv('D:/Study/wind_datasets/nwp_benchmarks/gefcom2012_wind/data/windpowermeasurements.csv')
            power=pd.Series(raw.wp1.to_numpy(),index=pd.to_datetime(raw.date.astype(str),format='%Y%m%d%H'))
        target_times=d['origins'][:,None]+np.arange(1,25)[None,:]*np.timedelta64(1,'h')
        history_times=d['origins'][:,None]+np.arange(-23,1)[None,:]*np.timedelta64(1,'h')
        same('raw_dataset_labels',d['y'],power.reindex(target_times.ravel()).to_numpy().reshape(d['y'].shape))
        same('raw_dataset_history',d['history'],power.reindex(history_times.ravel()).to_numpy().reshape(d['history'].shape))
        gt_cells+=d['y'].size
        weights=load_npz(f/'forecast_weights.npz');support=dist['support']
        sums=np.asarray(weights.sum(axis=1)).ravel();same('weights_total_mass',sums,np.ones(len(sums)))
        assert weights.data.min()>0
        for pos,day in enumerate(forecast):
          for lead in range(24):
            row=pos*24+lead;lo,hi=weights.indptr[row:row+2];x=support[weights.indices[lo:hi]];w=weights.data[lo:hi];w=w/w.sum();y=d['y'][day,lead]
            same('p_zero',np.sum(w[x<=.01]),dist['p_zero'][day,lead])
            if np.isfinite(y):same('randomized_PIT',np.clip(np.sum(w[x<y])+dist['uniform'][day,lead]*np.sum(w[x==y]),0,1),dist['pit'][day,lead])
            pit_cells+=1
        # Reconstruct all 500 bootstrap trees for a small real-forecast witness.
        model=joblib.load(ROOT/f'out_gefcom_revision_20261104/qrf/gefcom{year}/farm_01/leaf_10/seed_0/model.joblib')['model'];model.n_jobs=2
        assert model.bootstrap and model.max_samples is None and model.max_samples_leaf is None
        dense=zload(ROOT/f'out_gefcom_pooling_20261105/inputs/gefcom{year}/farm_01/features.npz')['QRF']
        trainids=np.flatnonzero(meta.phase.eq('train')&d['valid']);trainrows=(trainids[:,None]*24+np.arange(24)).ravel();xt=dense[trainrows];yt=d['y'][trainids].ravel()
        np.testing.assert_array_equal(np.unique(yt),support)
        choices=np.unique(np.linspace(0,len(forecast)*24-1,12,dtype=int));flatro=(forecast[:,None]*24+np.arange(24)).ravel();testx=dense[flatro[choices]]
        explicit=np.zeros((len(choices),len(support)))
        for tree in model.estimators_:
            bag=np.random.RandomState(tree.random_state).randint(0,len(yt),len(yt))
            trainleaf=tree.apply(xt)[bag];testleaf=tree.apply(testx)
            for j,leaf in enumerate(testleaf):
                responses=yt[bag[trainleaf==leaf]]
                explicit[j]+=np.bincount(np.searchsorted(support,responses),minlength=len(support))/(len(responses)*len(model.estimators_))
        same('explicit_500_tree_leaf_weights',explicit,weights[choices].toarray(),tol=4e-13)
        native=model.predict(testx,quantiles=Q.tolist(),weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True)
        same('native_linear_quantiles',native,d['pred'][forecast].reshape(-1,7)[choices])
        for j,row in enumerate(choices):
            day=forecast[row//24];lead=row%24;nz=explicit[j]>0;x=support[nz];w=explicit[j,nz]
            same('raw_ECDF_quantiles',quant(x,w,Q),dist['raw_ecdf_quantiles'][day,lead])
            same('raw_ECDF_pairwise_CRPS',crps_pair(x,w,d['y'][day,lead]),dist['raw_ecdf_crps'][day,lead])
        forest_cells+=len(choices)
        observed=[]
        for stage in ('validation','evaluation'):
            a=zload(f/f'{stage}.npz');boundary=zload(f/f'{stage}_boundary_probabilities.npz');ids=a['indices'];offset={int(day):i for i,day in enumerate(forecast)}
            choices={(int(pos),lead) for pos in np.unique(np.linspace(0,len(ids)-1,10,dtype=int)) for lead in (0,5,6,11,17,23)}
            for state in (0,1,2):
              for fallback in (True,False):
                loc=np.argwhere((states[ids]==state)&((a['lambdas']==0)==fallback))
                choices.update(tuple(map(int,row)) for row in loc[::max(1,len(loc)//4)][:4])
            for pos,lead in sorted(choices):
                idx=ids[pos];used=cal_ids(d,meta,ids,pos,stage);li,lh,nd,lam=pool(used,lead,states[idx,lead],states)
                assert nd==a['local_days'][pos,lead];same('state_lambda',lam,a['lambdas'][pos,lead])
                row=offset[int(idx)]*24+lead;start,end=weights.indptr[row:row+2];x=support[weights.indices[start:end]];w=weights.data[start:end]
                for arm,use_local in ((4,False),(5,True)):
                    values,prob=mix(dist['pit'][used,lead],dist['pit'][li,lh],lam if use_local else 0.)
                    law=quant(x,w,values,True)
                    same('PIT_pushforward_quantiles',quant(law,prob,Q),a['pred'][arm,pos,lead])
                    same('PIT_pairwise_CRPS',crps_pair(law,prob,d['y'][idx,lead]),a['crps'][arm,pos,lead])
                    bp=np.array([prob[law<=.01].sum(),prob[law==0].sum(),prob[law==1].sum()])
                    same('PIT_boundary_probabilities',bp,boundary[str(a['methods'][arm])][pos,lead])
                for arm,use_local in ((2,False),(6,True)):
                  for alpha,lq,uq in ((.2,2,4),(.1,1,5),(.05,0,6)):
                    rank=int(np.ceil((len(used)+1)*(1-alpha/2)-1e-12));threshold=rank/len(used)
                    sl=d['pred'][used,lead,lq]-d['y'][used,lead];su=d['y'][used,lead]-d['pred'][used,lead,uq]
                    lv,lw=mix(sl,d['pred'][li,lh,lq]-d['y'][li,lh],lam if use_local else 0.)
                    uv,uw=mix(su,d['y'][li,lh]-d['pred'][li,lh,uq],lam if use_local else 0.)
                    lower=d['pred'][idx,lead,lq]-quant(lv,lw,[threshold])[0];upper=d['pred'][idx,lead,uq]+quant(uv,uw,[threshold])[0]
                    if lower>upper:lower=upper=(d['pred'][idx,lead,lq]+d['pred'][idx,lead,uq])/2
                    same('DualTail_finite_ranks_collapse_clip',[np.clip(lower,0,1),np.clip(upper,0,1)],a['pred'][arm,pos,lead,[lq,uq]])
                observed.append({'stage':stage,'position':pos,'lead':lead+1,'state':int(states[idx,lead]),'local_days':nd,'lambda':lam})
                sample_cells+=1
        details.append({'year':year,'farm':farm,'real_forest_cells':12,'calibration_witnesses':observed})
        del model,weights,dense,xt,explicit;gc.collect()
        print('verified year',year,'farm1 full-law and raw-label witnesses',flush=True)
    R['counts']={'all_farm_baseline_values':total_base,'all_farm_arrival_days':total_days,'farm1_raw_label_cells':gt_cells,'farm1_raw_history_cells':gt_cells,'farm1_PIT_cells':pit_cells,'full_500_tree_forecast_cells':forest_cells,'PIT_and_DualTail_calibration_witness_cells':sample_cells}
    R['witnesses']=details;R['status']='pass';R['audited_hashes']={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'route_a/p1_distribution.py',ROOT/'route_a/p1_run.py',ROOT/'route_a/p1_analyze.py',RUN/'REPORT.md',RUN/'metrics_macro.csv',RUN/'scores_macro.csv',RUN/'joint_gates.csv',Path(__file__))}
    (OUT/'independent_distribution_verification.json').write_text(json.dumps(R,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in R.items() if k!='witnesses'},indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
