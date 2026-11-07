"""Fresh reviewer read-only checks; no optimizer steps or training reruns."""
from pathlib import Path
import hashlib, json, math, gc, os
from route_a.p2_models import torch, make_model, LEVELS
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[5]
RUN = ROOT / 'experiments/route_a/20261107_p2_v040'
P0 = ROOT / 'experiments/route_a/20261107_p0_v020'
OUT = Path(__file__).resolve().parent
INPUT = ROOT / 'out_gefcom_pooling_20261105/inputs'
LEVEL = np.asarray(LEVELS)

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def diff(a, b):
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))

def close(a, b, atol=1e-12):
    np.testing.assert_allclose(a, b, rtol=0, atol=atol)

def forward(m, x, batch=128):
    m.eval()
    with torch.no_grad():
        return np.concatenate([m(torch.from_numpy(x[i:i+batch])).numpy() for i in range(0,len(x),batch)])

def loss(m, x, y):
    # Same reduction precision as saved training losses, independently written.
    m.eval()
    with torch.no_grad():
        p = m(torch.from_numpy(x)); err=torch.from_numpy(y).unsqueeze(-1)-p
        q=torch.tensor(LEVELS, dtype=p.dtype)
        return float(torch.maximum(q*err, (q-1)*err).mean())

def verify():
    report={'scope':'fresh same-family provisional; artifact/code review and replay only, no training reruns',
            'guard':os.environ.get('ROUTE_A_GUARD'), 'all_models':[], 'representatives':[]}
    assert report['guard']=='1'
    manifest_checks=[]
    for name in ('training_code_manifest.json','input_manifest.json'):
        entries=read(RUN/name)
        for e in entries:
            p=ROOT/e['path']; assert sha(p)==e['sha256'],str(p)
            if 'bytes' in e: assert p.stat().st_size==e['bytes']
        manifest_checks.append({'name':name,'entries_verified':len(entries)})
    report['manifest_checks']=manifest_checks
    protocol=read(RUN/'protocol.json'); cfg=protocol['training']; contract=sha(RUN/'training_code_manifest.json')
    assert protocol==read(ROOT/'route_a/p2_protocol.json')
    summary=pd.read_csv(RUN/'training_summary.csv',float_precision='round_trip')
    curves=pd.read_csv(RUN/'learning_curves.csv',float_precision='round_trip')
    expected={(y,f,m,s) for y,n in [(2014,10),(2012,7)] for f in range(1,n+1) for m in ('MLP','TCN','GRU') for s in (0,1,2)}
    assert set(zip(summary.year,summary.farm,summary.model,summary.seed))==expected and len(summary)==153
    assert len(list(RUN.glob('gefcom*/farm_*/*/seed_*/model.pt')))==153
    assert len(list(RUN.glob('gefcom*/farm_*/*/seed_*/heads.npz')))==153
    loss_improved=0; all_curve_rows=0; selection_steps=0; refit_steps=0
    for row in summary.itertuples(index=False):
        folder=RUN/f'gefcom{row.year}/farm_{row.farm:02d}/{row.model}/seed_{row.seed}'
        fit=read(folder/'fit.json')
        assert fit['contract_hash']==contract
        for k,v in fit.items():
            assert getattr(row,k)==v,(folder,k,getattr(row,k),v)
        hist=pd.read_csv(folder/'selection_history.csv',float_precision='round_trip')
        refit=pd.read_csv(folder/'refit_history.csv',float_precision='round_trip')
        assert np.isfinite(hist.to_numpy()).all() and np.isfinite(refit.to_numpy()).all()
        assert np.array_equal(hist.epoch,np.arange(1,len(hist)+1))
        assert np.array_equal(refit.epoch,np.arange(1,len(refit)+1))
        best=float('inf'); epoch=0; stale=0
        for r in hist.itertuples():
            if r.stop_pinball < best-cfg['min_delta']: best=r.stop_pinball;epoch=r.epoch;stale=0
            else: stale+=1
            assert stale<cfg['patience'] or r.epoch==len(hist)
        assert epoch==fit['selected_epoch'] and best==fit['best_stop_pinball']
        assert len(hist)==fit['selection_epochs_run'] and len(refit)==epoch
        assert len(hist)==cfg['max_selection_epochs'] or stale==cfg['patience']
        assert fit['hit_epoch_cap']==(len(hist)==cfg['max_selection_epochs'])
        assert (hist.max_preclip_grad_norm>0).all() and (refit.max_preclip_grad_norm>0).all()
        for phase,part in [('selection',hist),('full_refit',refit)]:
            got=curves[(curves.year==row.year)&(curves.farm==row.farm)&(curves.model==row.model)&(curves.seed==row.seed)&(curves.phase==phase)]
            assert len(got)==len(part)
            # CSV aggregation can round at most an ulp when originally parsed without round_trip.
            close(got[list(part)].to_numpy(),part.to_numpy(),atol=1e-15)
            all_curve_rows+=len(part)
        ssteps=len(hist)*math.ceil(fit['optimization_days']/cfg['batch_origins'])
        rsteps=len(refit)*math.ceil(fit['full_training_days']/cfg['batch_origins'])
        selection_steps+=ssteps;refit_steps+=rsteps
        improved=fit['final_full_train_pinball']<fit['initial_full_train_pinball'];loss_improved+=int(improved)
        report['all_models'].append({'year':row.year,'farm':row.farm,'model':row.model,'seed':row.seed,
            'selected_epoch':epoch,'selection_epochs':len(hist),'selection_steps_implied_by_history':ssteps,
            'refit_steps_implied_by_history':rsteps,'loss_improved':improved,
            'minimum_logged_gradient':float(min(hist.max_preclip_grad_norm.min(),refit.max_preclip_grad_norm.min()))})
    assert all_curve_rows==len(curves)
    diag=summary.groupby(['year','model']).agg(models=('seed','size'),parameters_min=('parameters','min'),parameters_max=('parameters','max'),
        selected_epochs_mean=('selected_epoch','mean'),selected_epochs_min=('selected_epoch','min'),selected_epochs_max=('selected_epoch','max'),
        hit_epoch_cap=('hit_epoch_cap','sum'),initial_train_pinball=('initial_full_train_pinball','mean'),
        final_train_pinball=('final_full_train_pinball','mean'),total_seconds=('total_seconds','sum')).reset_index()
    saved=pd.read_csv(RUN/'training_diagnostics.csv',float_precision='round_trip')
    assert np.array_equal(diag[['year','model']],saved[['year','model']])
    close(diag.select_dtypes('number'),saved.select_dtypes('number'),atol=1e-12)
    report['all_model_summary']={'models':len(summary),'curve_rows':all_curve_rows,'loss_improved':loss_improved,
        'epoch_cap_hits':int(summary.hit_epoch_cap.sum()),'min_selected_epoch':int(summary.selected_epoch.min()),
        'max_selected_epoch':int(summary.selected_epoch.max()),'selection_optimizer_steps_implied':selection_steps,
        'refit_optimizer_steps_implied':refit_steps,'diagnostics_aggregate_max_error':diff(diag.select_dtypes('number'),saved.select_dtypes('number'))}
    for year in (2014,2012):
        p0=P0/f'gefcom{year}/farm_01'; folder=RUN/f'gefcom{year}/farm_01'; cached=INPUT/f'gefcom{year}/farm_01'
        with np.load(p0/'predictions.npz') as z: d={k:z[k] for k in z.files}
        meta=pd.read_csv(p0/'samples.csv')
        with np.load(folder/'indices.npz') as z: ix={k:z[k] for k in z.files}
        train=np.flatnonzero((meta.phase=='train')&d['valid']);cut=int(.8*len(train))
        expected_idx={'train':train,'optimization':train[:cut],'stop':train[cut:],'forecast':np.flatnonzero(meta.phase!='train'),'origins':d['origins']}
        for k,v in expected_idx.items(): np.testing.assert_array_equal(ix[k],v)
        assert np.intersect1d(ix['optimization'],ix['stop']).size==0
        assert np.intersect1d(ix['train'],ix['forecast']).size==0
        assert d['ends'][ix['optimization']].max()<=d['origins'][ix['stop']].min()
        assert d['ends'][ix['train']].max()<=d['origins'][ix['forecast']].min()
        with np.load(cached/'features.npz') as z: canonical=z['GBM'];dense=z['QRF']
        cells=lambda ids:(ids[:,None]*24+np.arange(24)).ravel()
        pre=joblib.load(folder/'preprocessing.joblib');original=joblib.load(cached/'preprocessing.joblib')['QRF_imputer']
        records={'year':year,'farm':1,'features':dense.shape[1],'train_days':len(train),'train_cells':len(train)*24,
            'optimization_days':cut,'stop_days':len(train)-cut,'forecast_days':len(ix['forecast']),
            'last_opt_label':str(d['ends'][ix['optimization']].max()),'first_stop_origin':str(d['origins'][ix['stop']].min()),
            'last_train_label':str(d['ends'][train].max()),'first_forecast_origin':str(d['origins'][ix['forecast']].min())}
        for stage,ids in [('selection',ix['optimization']),('final',train)]:
            imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(canonical[cells(ids)])
            np.testing.assert_equal(imputer.statistics_,pre[stage+'_imputer'].statistics_)
            np.testing.assert_equal(imputer.indicator_.features_,pre[stage+'_imputer'].indicator_.features_)
            transformed=imputer.transform(canonical)
            scinput=transformed if stage=='selection' else transformed.astype('float32')
            scaler=StandardScaler().fit(scinput[cells(ids)])
            for attr in ('mean_','var_','scale_','n_samples_seen_'):
                np.testing.assert_equal(getattr(scaler,attr),getattr(pre[stage+'_scaler'],attr))
            if stage=='final':
                np.testing.assert_equal(imputer.statistics_,original.statistics_)
                np.testing.assert_equal(imputer.indicator_.features_,original.indicator_.features_)
                np.testing.assert_array_equal(transformed.astype('float32'),dense)
            records[stage+'_preprocessor_exact']=True
        witness=read(folder/'data_contract.json')
        assert witness==next(x for x in read(RUN/'data_contracts.json') if x['year']==year and x['farm']==1)
        assert witness['train_label_cells']==len(train)*24 and witness['selection_imputer_fit_rows']==cut*24
        oldfit=read(ROOT/f'out_gefcom_revision_20261104/qrf/gefcom{year}/farm_01/leaf_10/seed_0/fit.json')
        assert oldfit['train_cells']==len(train)*24 and oldfit['features']==dense.shape[1]
        fullx=pre['final_scaler'].transform(dense).astype('float32').reshape(len(d['y']),24,-1)
        selectx=pre['selection_scaler'].transform(pre['selection_imputer'].transform(canonical)).astype('float32').reshape(len(d['y']),24,-1)
        y=d['y'].astype('float32'); records['models']=[]
        for name in ('MLP','TCN','GRU'):
            sf=folder/name/'seed_0';fit=read(sf/'fit.json')
            model=make_model(name,dense.shape[1],y[train],0)
            initial_state={k:v.clone() for k,v in model.state_dict().items()}
            initial_loss=loss(model,fullx[train],y[train]);close(initial_loss,fit['initial_full_train_pinball'],atol=0)
            repeated=make_model(name,dense.shape[1],y[train],0)
            assert all(torch.equal(initial_state[k],v) for k,v in repeated.state_dict().items())
            alternate=make_model(name,dense.shape[1],y[train],1)
            assert any(not torch.equal(initial_state[k],v) for k,v in alternate.state_dict().items())
            selection_model=make_model(name,dense.shape[1],y[ix['optimization']],0)
            close(loss(selection_model,selectx[ix['optimization']],y[ix['optimization']]),fit['initial_optimization_pinball'],atol=0)
            close(loss(selection_model,selectx[ix['stop']],y[ix['stop']]),fit['initial_stop_pinball'],atol=0)
            state=torch.load(sf/'model.pt',map_location='cpu',weights_only=True)
            changed=[k for k in state if not torch.equal(state[k],initial_state[k])]
            assert changed and all(torch.isfinite(v).all() for v in state.values())
            model.load_state_dict(state); assert sum(p.numel() for p in model.parameters())==fit['parameters']
            final_loss=loss(model,fullx[train],y[train]);close(final_loss,fit['final_full_train_pinball'],atol=0)
            pred=forward(model,fullx[ix['forecast']]).astype('float64')
            with np.load(sf/'heads.npz') as z:
                np.testing.assert_array_equal(z['quantiles'],LEVEL);saved_heads=z['prediction']
            np.testing.assert_array_equal(pred,saved_heads)
            assert np.isfinite(pred).all() and (np.diff(pred,axis=-1)>=0).all() and pred.min()>=0 and pred.max()<=1
            close(float((pred[:,:,0]==0).mean()),fit['raw_lower_zero_fraction'],atol=0)
            trace=[]
            for stage in ('validation','evaluation'):
                with np.load(sf/f'{stage}.npz') as z:
                    ids=np.flatnonzero(meta.phase==stage);np.testing.assert_array_equal(z['indices'],ids)
                    np.testing.assert_array_equal(z['pred'][0],pred[np.searchsorted(ix['forecast'],ids)])
                    for conf in (.8,.9,.95):
                        assert (z[f'{conf:.2f}_max_label_end']<=d['origins'][ids]).all()
                        assert (z[f'{conf:.2f}_n_cal']>0).all()
                    trace.append({'stage':stage,'origins':len(ids),'raw_replay_exact':True})
            records['models'].append({'name':name,'seed':0,'parameters':fit['parameters'],'initial_train_pinball':initial_loss,
                'final_train_pinball':final_loss,'checkpoint_prediction_max_error':diff(pred,saved_heads),
                'changed_parameter_tensors':len(changed),'all_parameter_tensors':len(state),
                'max_weight_change_from_same_seed_init':max(float((state[k]-initial_state[k]).abs().max()) for k in state),
                'seed_repeat_initialization_exact':True,'different_seed_initialization_distinct':True,
                'selection_initial_losses_exact':True,'stage_traces':trace})
            del model,repeated,alternate,selection_model,state,initial_state;gc.collect()
        qpack=joblib.load(p0/'refit.joblib');qm=qpack['model'];qm.n_jobs=2
        qids=ix['forecast'][[0,len(ix['forecast'])//2,-1]]
        qx=qpack['imputer'].transform(canonical)[cells(qids)]
        qp=qm.predict(qx,quantiles=LEVEL.tolist(),weighted_quantile=True,weighted_leaves=True,aggregate_leaves_first=True).reshape(len(qids),24,7)
        close(qp,d['pred'][qids],atol=1e-12)
        records['qrf_checkpoint_replay']={'origins':len(qids),'max_error':diff(qp,d['pred'][qids]),
            'n_estimators':qm.n_estimators,'min_samples_leaf':qm.min_samples_leaf,'seed':qm.random_state}
        for stage in ('validation','evaluation'):
            with np.load(folder/f'QRF/seed_0/{stage}.npz') as z, np.load(p0/f'{stage}_calibration.npz') as qz:
                ids=np.flatnonzero(meta.phase==stage);np.testing.assert_array_equal(z['indices'],ids)
                np.testing.assert_array_equal(z['pred'][0],d['pred'][ids])
                for conf in (.8,.9,.95):
                    prefix=f'QRF_CQR_rolling84_{conf:.2f}'
                    qi=int(np.argmin(abs(LEVEL-(1-conf)/2))); qj=int(np.argmin(abs(LEVEL-(1-(1-conf)/2))))
                    np.testing.assert_array_equal(z['pred'][1,:,:,qi],qz[prefix+'_lo'])
                    np.testing.assert_array_equal(z['pred'][1,:,:,qj],qz[prefix+'_hi'])
        records['qrf_stage_raw_and_calibrated_continuity_exact']=True
        report['representatives'].append(records)
        print('REPRESENTATIVE_VERIFIED',year,flush=True)
        del qpack,qm;gc.collect()
    report['status']='PASS'
    report['limits']=['No training rerun: nonzero gradients, changed weights, decreased losses and exact checkpoint replay corroborate optimization; optimizer state was not persisted.',
        'Detailed preprocessing/checkpoint/full prediction replay covers farm_01 in both datasets and neural seed0; 153 history contracts and seed metadata checked.',
        'Single fixed architecture per family, different normalization/recurrence and one frozen QRF seed prevent general family or pure temporal-structure causal claims.']
    (OUT/'training_subreview.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('all_models','representatives')},indent=2),flush=True)

if __name__=='__main__': verify()
