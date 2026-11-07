"""Same-information temporal baseline training; calibration is a separate stage."""
from pathlib import Path
import argparse,gc,json,time,platform
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write,new_run_dir
from route_a.p2_models import torch,make_model,pinball,eval_loss,predict,LEVELS
from route_a.p0_baseline import sha,write

P0=ROOT/'experiments/route_a/20261107_p0_v020'
INPUT=ROOT/'out_gefcom_pooling_20261105/inputs'


def train_epoch(model,opt,x,y,generator,batch_size):
    model.train();order=torch.randperm(len(x),generator=generator);total=0.;max_gradient=0.
    for start in range(0,len(x),batch_size):
        ids=order[start:start+batch_size];opt.zero_grad(set_to_none=True)
        loss=pinball(model(x[ids]),y[ids]);assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        assert torch.isfinite(norm);max_gradient=max(max_gradient,float(norm));opt.step()
        total+=float(loss.detach())*len(ids)
    return total/len(x),max_gradient


def fit_one(name,seed,x_opt,y_opt,x_stop,y_stop,x_full,y_full,x_forecast,folder,protocol,contract):
    summary_path=folder/'fit.json'
    if summary_path.exists() and (folder/'heads.npz').exists() and (folder/'model.pt').exists():
        summary=json.loads(summary_path.read_text(encoding='utf-8'))
        assert summary['contract_hash']==contract and summary['model']==name and summary['seed']==seed
        return summary
    folder.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();cfg=protocol['training']
    xx,yy=torch.from_numpy(x_opt),torch.from_numpy(y_opt)
    sx,sy=torch.from_numpy(x_stop),torch.from_numpy(y_stop)
    model=make_model(name,x_opt.shape[-1],y_opt,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed+104729)
    initial_train=eval_loss(model,xx,yy);initial_stop=eval_loss(model,sx,sy)
    best=float('inf');best_epoch=0;stale=0;history=[]
    for epoch in range(1,cfg['max_selection_epochs']+1):
        train_loss,gradient=train_epoch(model,opt,xx,yy,generator,cfg['batch_origins'])
        stop_loss=eval_loss(model,sx,sy);assert np.isfinite(stop_loss)
        history.append(dict(epoch=epoch,train_pinball=train_loss,stop_pinball=stop_loss,max_preclip_grad_norm=gradient))
        if stop_loss<best-cfg['min_delta']:
            best=stop_loss;best_epoch=epoch;stale=0
        else:stale+=1
        if stale>=cfg['patience']:break
    assert best_epoch>=1
    pd.DataFrame(history).to_csv(folder/'selection_history.csv',index=False)
    selection_seconds=time.perf_counter()-t0
    del model,opt,xx,yy,sx,sy;gc.collect()
    # Refit from a new initialization, not from the epoch-selection checkpoint.
    model=make_model(name,x_full.shape[-1],y_full,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    generator=torch.Generator().manual_seed(seed+104729)
    xx,yy=torch.from_numpy(x_full),torch.from_numpy(y_full)
    initial_full=eval_loss(model,xx,yy);refit=[]
    for epoch in range(1,best_epoch+1):
        loss,gradient=train_epoch(model,opt,xx,yy,generator,cfg['batch_origins'])
        refit.append(dict(epoch=epoch,train_pinball=loss,max_preclip_grad_norm=gradient))
    final_train=eval_loss(model,xx,yy)
    pred=predict(model,x_forecast)
    assert np.isfinite(pred).all() and (np.diff(pred,axis=-1)>=0).all() and pred.min()>=0 and pred.max()<=1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    np.savez_compressed(folder/'heads.npz',prediction=pred,quantiles=np.array(LEVELS))
    torch.save(model.state_dict(),folder/'model.pt')
    pd.DataFrame(refit).to_csv(folder/'refit_history.csv',index=False)
    summary=dict(model=name,seed=seed,contract_hash=contract,torch=torch.__version__,features=x_full.shape[-1],
        parameters=sum(p.numel() for p in model.parameters()),optimization_days=len(x_opt),stop_days=len(x_stop),
        full_training_days=len(x_full),forecast_days=len(x_forecast),selected_epoch=best_epoch,
        selection_epochs_run=len(history),hit_epoch_cap=len(history)==cfg['max_selection_epochs'],
        best_stop_pinball=best,initial_optimization_pinball=initial_train,initial_stop_pinball=initial_stop,
        initial_full_train_pinball=initial_full,final_full_train_pinball=final_train,
        selection_seconds=selection_seconds,total_seconds=time.perf_counter()-t0,
        raw_crossings=int((np.diff(pred,axis=-1)<0).sum()),raw_lower_zero_fraction=float((pred[:,:,0]==0).mean()),
        initialization='independent train-only empirical-quantile bias; full refit reinitializes at original seed',
        selection_labels='proper-training final20% only; never calibration/validation/evaluation',
        device='cpu',threads=torch.get_num_threads())
    write(summary_path,summary);del model,opt,xx,yy;gc.collect()
    return summary


def prepare_farm(year,farm,out):
    folder=out/f'gefcom{year}/farm_{farm:02d}';folder.mkdir(parents=True,exist_ok=True)
    prior=P0/f'gefcom{year}/farm_{farm:02d}';cached=INPUT/f'gefcom{year}/farm_{farm:02d}'
    with np.load(prior/'predictions.npz') as z:d={k:z[k] for k in ('y','valid','origins','ends')}
    meta=pd.read_csv(prior/'samples.csv');train=np.flatnonzero(meta.phase.eq('train')&d['valid'])
    cut=int(.8*len(train));optimization,stop=train[:cut],train[cut:]
    forecast=np.flatnonzero(~meta.phase.eq('train'))
    assert d['ends'][optimization].max()<=d['origins'][stop].min()
    assert d['ends'][train].max()<=d['origins'][forecast].min()
    with np.load(cached/'features.npz') as z:canonical=z['GBM'];dense=z['QRF']
    all_cells=np.arange(len(d['y'])*24).reshape(-1,24)
    tr_cells=all_cells[train].ravel();op_cells=all_cells[optimization].ravel()
    imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(canonical[tr_cells])
    original=joblib.load(cached/'preprocessing.joblib')['QRF_imputer']
    np.testing.assert_equal(imputer.statistics_,original.statistics_)
    np.testing.assert_array_equal(imputer.indicator_.features_,original.indicator_.features_)
    np.testing.assert_array_equal(imputer.transform(canonical).astype('float32'),dense)
    select_imputer=SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True).fit(canonical[op_cells])
    select_dense=select_imputer.transform(canonical)
    assert select_dense.shape[1]==dense.shape[1]
    select_scaler=StandardScaler().fit(select_dense[op_cells]);final_scaler=StandardScaler().fit(dense[tr_cells])
    select_x=select_scaler.transform(select_dense).astype('float32').reshape(len(d['y']),24,-1)
    full_x=final_scaler.transform(dense).astype('float32').reshape(len(d['y']),24,-1)
    assert np.isfinite(select_x).all() and np.isfinite(full_x).all() and np.isfinite(d['y'][train]).all()
    joblib.dump(dict(selection_imputer=select_imputer,selection_scaler=select_scaler,final_imputer=imputer,
        final_scaler=final_scaler),folder/'preprocessing.joblib')
    witness=dict(year=year,farm=farm,features=dense.shape[1],full_train_days=len(train),optimization_days=len(optimization),
        stop_days=len(stop),first_train_origin=str(d['origins'][train[0]]),last_train_label_end=str(d['ends'][train].max()),
        last_optimization_label_end=str(d['ends'][optimization].max()),first_stop_origin=str(d['origins'][stop].min()),
        first_forecast_origin=str(d['origins'][forecast].min()),QRF_imputer_exact=True,QRF_dense_features_exact=True,
        train_label_cells=int(len(train)*24),selection_imputer_fit_rows=int(len(op_cells)),final_scaler_fit_rows=int(len(tr_cells)))
    write(folder/'data_contract.json',witness)
    np.savez_compressed(folder/'indices.npz',train=train,optimization=optimization,stop=stop,forecast=forecast,origins=d['origins'])
    inputs=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size)
        for path in (prior/'predictions.npz',prior/'samples.csv',cached/'features.npz',cached/'preprocessing.joblib')]
    arrays=(select_x[optimization],d['y'][optimization].astype('float32'),select_x[stop],d['y'][stop].astype('float32'),
        full_x[train],d['y'][train].astype('float32'),full_x[forecast])
    return folder,arrays,witness,inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--out');p.add_argument('--resume',action='store_true');args=p.parse_args()
    out=require_local_write(ROOT/args.out) if args.out else new_run_dir('p2_temporal')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'training_completion.json').exists()
    protocol=json.loads((ROOT/'route_a/p2_protocol.json').read_text(encoding='utf-8'))
    source=['route_a/p2_protocol.json','route_a/p2_models.py','route_a/p2_train.py','requirements-route-a-temporal.txt']
    manifest=[dict(path=f,sha256=sha(ROOT/f)) for f in source]
    if (out/'protocol.json').exists():
        assert args.resume and json.loads((out/'protocol.json').read_text(encoding='utf-8'))==protocol
        assert json.loads((out/'training_code_manifest.json').read_text(encoding='utf-8'))==manifest
    else:write(out/'protocol.json',protocol);write(out/'training_code_manifest.json',manifest)
    contract=sha(out/'training_code_manifest.json');started=time.perf_counter();fits=[];inputs=[];data=[]
    write(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
        device='cpu',threads=2,CUDA_available=torch.cuda.is_available(),interpreter=str(ROOT/'.venv/Scripts/python.exe')))
    print('P2_OUTPUT',out,flush=True)
    for year,nfarms in ((2014,10),(2012,7)):
        for farm in range(1,nfarms+1):
            folder,arrays,witness,paths=prepare_farm(year,farm,out);inputs.extend(paths);data.append(witness)
            for name in protocol['networks']:
                for seed in protocol['seeds']:
                    summary=fit_one(name,seed,*arrays,folder/name/f'seed_{seed}',protocol,contract)
                    fits.append(dict(year=year,farm=farm,**summary))
                    print('FIT_DONE',year,farm,name,seed,'epoch',summary['selected_epoch'],'seconds',round(summary['total_seconds'],2),flush=True)
            del arrays;gc.collect()
            pd.DataFrame(fits).to_csv(out/'training_summary.csv',index=False)
            write(out/'input_manifest.json',inputs);write(out/'data_contracts.json',data)
    write(out/'training_completion.json',dict(status='complete',version='0.4.0',farms=17,architectures=protocol['networks'],
        seeds=protocol['seeds'],final_models=len(fits),selection_passes=len(fits),full_training_refits=len(fits),
        QRF_refits=0,elapsed_seconds=time.perf_counter()-started,
        scope='historical data, fixed architecture/optimizer budget; no calibration/evaluation labels used for stopping/fitting'))
    print('P2_TRAINING_COMPLETE',out,round(time.perf_counter()-started,2),flush=True)


if __name__=='__main__':main()
