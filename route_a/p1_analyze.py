"""Fixed gates, boundary diagnostics, and artifact checks for a completed P1."""
from pathlib import Path
import argparse,json,platform,subprocess
import numpy as np
import pandas as pd
import sklearn,quantile_forest
from scipy.sparse import load_npz
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p0_baseline import sha,write
from route_a.p1_run import P0,ARMS,REFERENCE
from route_a.p1_distribution import arrived,local_pool,mixed_values


def gates(run):
    m=pd.read_csv(run/'metrics_macro.csv');s=pd.read_csv(run/'scores_macro.csv');rows=[]
    m=m[m.confidence.eq(.9)&m.dimension.eq('regime')].set_index(['year','stage','method','group'])
    s=s.set_index(['year','stage','method','group'])
    for year in (2014,2012):
        for stage in ('validation','evaluation'):
            base=m.loc[year,stage,REFERENCE,'all'];bs=s.loc[year,stage,REFERENCE,'all']
            for method in ARMS:
                a=m.loc[year,stage,method,'all'];ss=s.loc[year,stage,method,'all']
                up=m.loc[year,stage,method,'up_ramp'];bu=m.loc[year,stage,REFERENCE,'up_ramp']
                zero=m.loc[year,stage,method,'near_zero'];bz=m.loc[year,stage,REFERENCE,'near_zero']
                checks=dict(coverage=float(a.coverage)>=.9,coverage_drop=float(a.coverage)>=float(base.coverage)-.005,
                    width=float(a.width)<=1.01*float(base.width),IS=float(a.IS)<=float(base.IS),WIS=float(ss.WIS)<=float(bs.WIS),
                    up_coverage=float(up.coverage)>=float(bu.coverage)+.01,
                    zero_coverage=float(zero.coverage)>=float(bz.coverage)+.01,
                    up_IS=float(up.IS)<=float(bu.IS),zero_IS=float(zero.IS)<=float(bz.IS))
                rows.append(dict(year=year,stage=stage,method=method,eligible=method in ('DualTail84','PIT84','PIT84_state','DualTail84_state'),
                    coverage=float(a.coverage),width_ratio=float(a.width/base.width),IS_ratio=float(a.IS/base.IS),WIS_ratio=float(ss.WIS/bs.WIS),
                    common_median_WIS_ratio=float(ss.common_median_WIS/bs.common_median_WIS),
                    up_coverage_gain=float(up.coverage-bu.coverage),zero_coverage_gain=float(zero.coverage-bz.coverage),
                    up_IS_ratio=float(up.IS/bu.IS),zero_IS_ratio=float(zero.IS/bz.IS),
                    joint_pass=bool(all(checks.values())),**{'pass_'+k:bool(v) for k,v in checks.items()}))
    df=pd.DataFrame(rows);df.to_csv(run/'joint_gates.csv',index=False)
    dev=df[df.year.eq(2014)&df.stage.eq('validation')&df.eligible&df.joint_pass]
    selected=REFERENCE if dev.empty else dev.sort_values(['WIS_ratio','method']).iloc[0]['method']
    write(run/'selection.json',dict(source='2014 validation only, fixed joint gate; all dates retrospective',
        eligible_passing=dev.method.tolist(),selected=selected,
        rule='among passing candidates minimize validation WIS; otherwise retain CQR_signed84',evaluation_used_for_selection=False))


def boundary_checks(run):
    rows=[]
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            folder=run/f'gefcom{year}/farm_{farm:02d}';prior=P0/f'gefcom{year}/farm_{farm:02d}'
            with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in ('y','valid','origins','ends')}
            meta=pd.read_csv(prior/'samples.csv')
            with np.load(folder/'distribution_inputs.npz') as z:
                support,pit,states,forecast,pzero=z['support'],z['pit'],z['states'],z['forecast'],z['p_zero']
            weights=load_npz(folder/'forecast_weights.npz')
            f0=np.full(d['y'].shape,np.nan);f1minus=f0.copy()
            f0[forecast]=np.asarray(weights[:,support==0].sum(axis=1)).reshape(-1,24)
            f1minus[forecast]=1-np.asarray(weights[:,support==1].sum(axis=1)).reshape(-1,24)
            for stage in ('validation','evaluation'):
                ids=np.flatnonzero(meta.phase.eq(stage));seed=np.flatnonzero(meta.phase.eq('validation_seed' if stage=='validation' else 'validation'))
                probs={method:np.empty((len(ids),24,3)) for method in ('QRF_ECDF_raw','PIT84','PIT84_state')}
                probs['QRF_ECDF_raw']=np.stack([pzero[ids],f0[ids],1-f1minus[ids]],axis=-1)
                for pos,idx in enumerate(ids):
                    used=arrived(d,seed,ids,pos)
                    for lead in range(24):
                        local_i,local_h,weight,_=local_pool(used,lead,states[idx,lead],states)
                        for method,shrink in (('PIT84',False),('PIT84_state',True)):
                            v,w=mixed_values(pit[used,lead],pit[local_i,local_h],weight if shrink else 0.)
                            # Pushforward law maps PIT0->physical0 and PIT1->physical1.
                            probs[method][pos,lead]=[
                                w[(v<1)&(v<=pzero[idx,lead]+1e-14)].sum(),
                                w[(v<1)&(v<=f0[idx,lead]+1e-14)].sum(),
                                1-w[(v<1)&(v<=f1minus[idx,lead]+1e-14)].sum()]
                good=np.broadcast_to(d['valid'][ids,None],(len(ids),24));event=(d['y'][ids]<=.01).astype(float)
                for method,p in probs.items():
                    assert np.isfinite(p).all() and p.min()>=-1e-12 and p.max()<=1+1e-12
                    p=np.clip(p,0.,1.);probs[method]=p
                    rows.append(dict(year=year,farm=farm,stage=stage,method=method,cells=int(good.sum()),
                        near_zero_Brier=float(np.square(p[:,:,0]-event)[good].mean()),
                        mean_predicted_near_zero=float(p[:,:,0][good].mean()),observed_near_zero=float(event[good].mean()),
                        mean_zero_mass=float(p[:,:,1][good].mean()),mean_one_mass=float(p[:,:,2][good].mean())))
                np.savez_compressed(folder/f'{stage}_boundary_probabilities.npz',indices=ids,**probs)
    pd.DataFrame(rows).to_csv(run/'boundary_metrics.csv',index=False)


def verify(run):
    checks=json.loads((run/'forest_checks.json').read_text(encoding='utf-8'))
    prior={Path(e['path']).as_posix():e['sha256'] for e in json.loads((P0/'input_manifest.json').read_text(encoding='utf-8'))}
    for w in checks:
        path=f"out_gefcom_revision_20261104/qrf/gefcom{w['year']}/farm_{w['farm']:02d}/leaf_10/seed_0/model.joblib"
        assert w['forest_sha256_before']==w['forest_sha256_after']==prior[path]==sha(ROOT/path)
    for e in json.loads((run/'input_manifest.json').read_text(encoding='utf-8')):
        assert sha(ROOT/e['path'])==e['sha256'],e['path']
    for e in json.loads((run/'code_manifest.json').read_text(encoding='utf-8')):
        assert sha(ROOT/e['path'])==e['sha256'],e['path']
    p0=pd.read_csv(P0/'metrics_macro.csv');p1=pd.read_csv(run/'metrics_macro.csv')
    errors=[]
    for old,new in [('QRF_raw','QRF_linear_raw'),('QRF_CQR_rolling84','CQR_signed84')]:
        keys=['year','stage','confidence','dimension','group'];cols=['coverage','width','IS','lower_miss','upper_miss']
        a=p0[p0.method.eq(old)].set_index(keys)[cols].sort_index();b=p1[p1.method.eq(new)].set_index(keys)[cols].sort_index()
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12);errors.append(float(abs(a-b).to_numpy().max()))
    return dict(status='pass',forest_hashes_match_P0=True,new_forests_fit=0,farms=len(checks),
        baseline_macro_max_error=max(errors),max_dualtail90_legacy_error=max(max(w['validation_dual90_legacy_max_error'],w['evaluation_dual90_legacy_max_error']) for w in checks),
        all_input_and_execution_code_hashes_match=True,analyzer_sha256=sha(Path(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=require_local_write(ROOT/args.run)
    assert json.loads((run/'completion.json').read_text(encoding='utf-8'))['status']=='complete'
    checks=verify(run);gates(run);boundary_checks(run)
    records={}
    for name,script in [('isolation','tools/verify_isolation.py'),('PIT_and_dualtail_units','route_a/check_p1.py')]:
        p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'tools/run_route_a.py','--script',script],capture_output=True,text=True,encoding='utf-8')
        records[name]=dict(exit_code=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip());assert p.returncode==0
    write(run/'verification_checks.json',records);write(run/'release_checks.json',checks)
    write(run/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
        quantile_forest=quantile_forest.__version__,interpreter=str(ROOT/'.venv/Scripts/python.exe'),new_dependency_installs=0,CPU_threads=2))
    write(run/'analysis_completion.json',dict(status='complete',source_sha256=sha(Path(__file__)),full_CDF_boundary_scores=True))
    print('P1_ANALYSIS_COMPLETE',run)


if __name__=='__main__':main()
