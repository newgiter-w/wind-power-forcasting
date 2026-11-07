"""Training-budget checks, frozen gates and independent checkpoint replay for P2."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
import pandas as pd
import joblib
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p0_baseline import sha,write
from route_a.p2_models import torch,make_model,predict
from route_a.p2_train import P0,INPUT


def make_gates(metrics,scores,run,by_seed=False):
    rows=[];m=metrics[metrics.confidence.eq(.9)&metrics.dimension.eq('regime')]
    for year in (2014,2012):
        for stage in ('validation','evaluation'):
            ref=m[m.year.eq(year)&m.stage.eq(stage)&m.method.eq('QRF_CQR')].set_index('group')
            rs=scores[scores.year.eq(year)&scores.stage.eq(stage)&scores.method.eq('QRF_CQR')].set_index('group')
            for method in ('MLP_CQR','TCN_CQR','GRU_CQR'):
                for seed in ((0,1,2) if by_seed else (-1,)):
                    a=m[m.year.eq(year)&m.stage.eq(stage)&m.method.eq(method)]
                    ss=scores[scores.year.eq(year)&scores.stage.eq(stage)&scores.method.eq(method)]
                    if by_seed:a=a[a.seed.eq(seed)];ss=ss[ss.seed.eq(seed)]
                    a=a.set_index('group');ss=ss.set_index('group')
                    target=a.loc['all'];baseline=ref.loc['all']
                    checks=dict(coverage=target.coverage>=.9,coverage_drop=target.coverage>=baseline.coverage-.005,
                        width=target.width<=1.01*baseline.width,IS=target.IS<=baseline.IS,
                        WIS=ss.loc['all','WIS']<=rs.loc['all','WIS'],
                        up_coverage=a.loc['up_ramp','coverage']>=ref.loc['up_ramp','coverage']+.01,
                        zero_coverage=a.loc['near_zero','coverage']>=ref.loc['near_zero','coverage']+.01,
                        up_IS=a.loc['up_ramp','IS']<=ref.loc['up_ramp','IS'],
                        zero_IS=a.loc['near_zero','IS']<=ref.loc['near_zero','IS'])
                    rows.append(dict(year=year,stage=stage,method=method,seed=seed,coverage=float(target.coverage),
                        width_ratio=float(target.width/baseline.width),IS_ratio=float(target.IS/baseline.IS),
                        WIS_ratio=float(ss.loc['all','WIS']/rs.loc['all','WIS']),
                        up_coverage_gain=float(a.loc['up_ramp','coverage']-ref.loc['up_ramp','coverage']),
                        zero_coverage_gain=float(a.loc['near_zero','coverage']-ref.loc['near_zero','coverage']),
                        up_IS_ratio=float(a.loc['up_ramp','IS']/ref.loc['up_ramp','IS']),
                        zero_IS_ratio=float(a.loc['near_zero','IS']/ref.loc['near_zero','IS']),
                        joint_pass=bool(all(checks.values())),**{'pass_'+k:bool(v) for k,v in checks.items()}))
    df=pd.DataFrame(rows);df.to_csv(run/('joint_gates_by_seed.csv' if by_seed else 'joint_gates.csv'),index=False)
    if not by_seed:
        dev=df[df.year.eq(2014)&df.stage.eq('validation')&df.joint_pass]
        selected='QRF_CQR' if dev.empty else str(dev.sort_values(['WIS_ratio','method']).iloc[0]['method'])
        write(run/'selection.json',dict(source='2014 validation neural-seed-mean only; historical development data',
            eligible_passing=dev.method.tolist(),selected=selected,
            rule='minimize seed-mean WIS among joint-gate-passing calibrated candidates; otherwise retain QRF_CQR',
            seed_selection=False,ensemble=False,evaluation_used_for_selection=False))


def verify_training(run):
    protocol=json.loads((run/'protocol.json').read_text(encoding='utf-8'));cfg=protocol['training']
    fits=pd.read_csv(run/'training_summary.csv');assert len(fits)==153
    assert set(fits.model)=={'MLP','TCN','GRU'} and set(fits.seed)=={0,1,2}
    old_inputs={Path(e['path']).as_posix():e for e in json.loads((P0/'input_manifest.json').read_text(encoding='utf-8'))}
    for e in json.loads((run/'input_manifest.json').read_text(encoding='utf-8')):
        assert sha(ROOT/e['path'])==e['sha256']
    for name in ('training_code_manifest.json','evaluation_code_manifest.json'):
        for e in json.loads((run/name).read_text(encoding='utf-8')):assert sha(ROOT/e['path'])==e['sha256'],e['path']
    max_error=0.;ranks_checked=0;model_hashes=[];curves=[]
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            prior=P0/f'gefcom{year}/farm_{farm:02d}';folder=run/f'gefcom{year}/farm_{farm:02d}'
            model_path=f'out_gefcom_revision_20261104/qrf/gefcom{year}/farm_{farm:02d}/leaf_10/seed_0/model.joblib'
            assert sha(ROOT/model_path)==old_inputs[model_path]['sha256']
            model_hashes.append(dict(year=year,farm=farm,QRF_sha256=old_inputs[model_path]['sha256']))
            with np.load(prior/'predictions.npz') as z:y=z['y'];origins=z['origins'];ends=z['ends']
            with np.load(folder/'indices.npz') as z:train=z['train'];optimization=z['optimization'];stop=z['stop'];forecast=z['forecast']
            assert ends[optimization].max()<=origins[stop].min() and ends[train].max()<=origins[forecast].min()
            fit_original=json.loads((ROOT/model_path).with_name('fit.json').read_text(encoding='utf-8'))
            assert fit_original['train_cells']==len(train)*24
            with np.load(INPUT/f'gefcom{year}/farm_{farm:02d}/features.npz') as z:dense=z['QRF']
            preprocessing=joblib.load(folder/'preprocessing.joblib')
            expected=dense[(train[:,None]*24+np.arange(24)).ravel()].mean(axis=0,dtype='float64')
            np.testing.assert_allclose(preprocessing['final_scaler'].mean_,expected,rtol=0,atol=1e-10)
            # Match the production inference batch; checkpoint replay is not refitting.
            x=preprocessing['final_scaler'].transform(dense).astype('float32').reshape(len(y),24,-1)[forecast[:128]]
            for name in ('MLP','TCN','GRU'):
                for seed in (0,1,2):
                    sf=folder/name/f'seed_{seed}';summary=json.loads((sf/'fit.json').read_text(encoding='utf-8'))
                    history=pd.read_csv(sf/'selection_history.csv',float_precision='round_trip')
                    best=float('inf');epoch=0;stale=0
                    for row in history.itertuples():
                        if row.stop_pinball<best-cfg['min_delta']:best=float(row.stop_pinball);epoch=int(row.epoch);stale=0
                        else:stale+=1
                        assert stale<cfg['patience'] or row.epoch==len(history)
                    assert epoch==summary['selected_epoch'] and best==summary['best_stop_pinball']
                    assert len(history)==cfg['max_selection_epochs'] or stale==cfg['patience']
                    refit=pd.read_csv(sf/'refit_history.csv');assert len(refit)==epoch
                    curves.append(history.assign(year=year,farm=farm,model=name,seed=seed,phase='selection'))
                    curves.append(refit.assign(year=year,farm=farm,model=name,seed=seed,phase='full_refit'))
                    model=make_model(name,dense.shape[1],y[train].astype('float32'),seed)
                    model.load_state_dict(torch.load(sf/'model.pt',map_location='cpu',weights_only=True))
                    pred=predict(model,x)
                    with np.load(sf/'heads.npz') as z:error=float(abs(pred-z['prediction'][:len(x)]).max())
                    assert error<2e-6,(year,farm,name,seed,error)
                    max_error=max(max_error,error);ranks_checked+=1
    write(run/'forest_hash_continuity.json',model_hashes)
    pd.concat(curves,ignore_index=True).to_csv(run/'learning_curves.csv',index=False)
    return dict(status='pass',models=153,selection_epoch_rules_verified=ranks_checked,
        checkpoint_replays=ranks_checked,checkpoint_replay_max_error=max_error,
        identical_QRF_training_label_budget=True,all_source_and_input_hashes_match=True,QRF_forests_unchanged=17,
        internal_split_chronology=True,scaler_full_proper_training_only=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();run=require_local_write(ROOT/args.run)
    assert json.loads((run/'completion.json').read_text(encoding='utf-8'))['status']=='complete'
    m=pd.read_csv(run/'metrics_macro.csv');s=pd.read_csv(run/'scores_macro.csv')
    make_gates(m,s,run)
    make_gates(pd.read_csv(run/'metrics_seed_macro.csv'),pd.read_csv(run/'scores_seed_macro.csv'),run,True)
    verification=verify_training(run)
    p0=pd.read_csv(P0/'metrics_macro.csv');columns=['coverage','width','IS','lower_miss','upper_miss'];keys=['year','stage','confidence','dimension','group']
    errors=[]
    for new,old in [('QRF_raw','QRF_raw'),('QRF_CQR','QRF_CQR_rolling84')]:
        a=m[m.method.eq(new)].set_index(keys)[columns].sort_index();b=p0[p0.method.eq(old)].set_index(keys)[columns].sort_index()
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-12);errors.append(float(abs(a-b).to_numpy().max()))
    verification['baseline_macro_max_error']=max(errors)
    write(run/'release_checks.json',verification)
    fit=pd.read_csv(run/'training_summary.csv')
    fit.groupby(['year','model']).agg(models=('seed','size'),parameters_min=('parameters','min'),parameters_max=('parameters','max'),
        selected_epochs_mean=('selected_epoch','mean'),selected_epochs_min=('selected_epoch','min'),selected_epochs_max=('selected_epoch','max'),
        hit_epoch_cap=('hit_epoch_cap','sum'),initial_train_pinball=('initial_full_train_pinball','mean'),
        final_train_pinball=('final_full_train_pinball','mean'),total_seconds=('total_seconds','sum')).reset_index().to_csv(run/'training_diagnostics.csv',index=False)
    write(run/'analysis_completion.json',dict(status='complete',source_sha256=sha(Path(__file__)),gates='predeclared joint gate, neural seed mean; per-seed diagnostics also saved'))
    print('P2_ANALYSIS_COMPLETE',run)


if __name__=='__main__':main()
