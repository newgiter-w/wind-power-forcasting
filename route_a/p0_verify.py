"""Deterministic artifact consistency checks, separate from semantic peer review."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p0_baseline import sha,write,Q,ALPHAS,METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=require_local_write(ROOT/args.run)
    inputs=json.loads((run/'input_manifest.json').read_text())
    for item in inputs:
        path=Path(item['path']);path=path if path.is_absolute() else ROOT/path
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
    for item in json.loads((run/'code_manifest.json').read_text()):
        assert sha(ROOT/item['path'])==item['sha256'],item['path']
    diagnosis=json.loads((run/'diagnosis_completion.json').read_text())
    assert diagnosis['source_sha256']==sha(ROOT/'route_a/p0_diagnose.py')
    s=pd.read_csv(run/'scores_per_farm.csv');pin=pd.read_csv(run/'pinball_per_farm.csv')
    raw=s[s.method.eq(METHODS[0])].set_index(['year','farm','stage']).WIS
    identity=2*pin.groupby(['year','farm','stage']).pinball.mean()
    np.testing.assert_allclose(raw.sort_index(),identity.sort_index(),rtol=0,atol=1e-14)
    m=pd.read_csv(run/'metrics_per_farm.csv');maximum=0.;endpoint_files=0
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            folder=run/f'gefcom{year}/farm_{farm:02d}'
            with np.load(folder/'predictions.npz') as z:
                pred,y,valid,origins=z['pred'],z['y'],z['valid'],z['origins']
            for stage in ('validation','evaluation'):
                with np.load(folder/f'{stage}_calibration.npz') as c:
                    ids=c['indices'];truth=y[ids];good=np.broadcast_to(valid[ids,None],truth.shape)
                    for alpha in ALPHAS:
                        qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
                        for method in METHODS:
                            if method==METHODS[0]:lo,hi=pred[ids,:,qi],pred[ids,:,qj]
                            else:
                                prefix=f'{method}_{1-alpha:.2f}'
                                lo,hi=c[prefix+'_lo'],c[prefix+'_hi']
                                assert (c[prefix+'_max_label_end']<=origins[ids]).all()
                            assert (lo<=hi).all() and np.isfinite(lo).all() and np.isfinite(hi).all()
                            r=m[m.year.eq(year)&m.farm.eq(farm)&m.stage.eq(stage)&m.method.eq(method)&m.confidence.eq(1-alpha)&m.dimension.eq('regime')&m.group.eq('all')].iloc[0]
                            err_low=np.maximum(lo-truth,0);err_high=np.maximum(truth-hi,0)
                            vals={'coverage':((truth>=lo)&(truth<=hi))[good].mean(),'width':(hi-lo)[good].mean(),
                                'IS':(hi-lo+2/alpha*(err_low+err_high))[good].mean()}
                            for key,value in vals.items():
                                error=abs(float(r[key])-float(value));maximum=max(maximum,error)
                                assert error<1e-12,(year,farm,stage,method,alpha,key)
                    endpoint_files+=1
    witnesses=json.loads((run/'reproduction.json').read_text())
    assert len(witnesses)==17 and all(w['raw_max_abs_error']==0 for w in witnesses)
    assert sum('refit_max_abs_error' in w for w in witnesses)==2
    assert all(w.get('refit_max_abs_error',0)==0 for w in witnesses)
    checks=dict(status='pass',type='deterministic artifact checks, not semantic assurance',
        source_and_input_hashes_verified=True,input_files_checked=len(inputs),
        frozen_replays=17,from_scratch_refits=2,endpoint_files_checked=endpoint_files,
        all_overall_metric_recomputation_max_error=maximum,
        raw_WIS_equals_twice_mean_pinball_max_error=float(abs(raw.sort_index()-identity.sort_index()).max()),
        all_calibration_label_times_legal=True,diagnosis_source_hash=sha(ROOT/'route_a/p0_diagnose.py'),
        verifier_source_hash=sha(Path(__file__)))
    write(run/'release_checks.json',checks)
    print(json.dumps(checks))


if __name__=='__main__':main()
