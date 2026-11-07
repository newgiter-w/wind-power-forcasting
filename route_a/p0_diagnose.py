"""Supplementary read-only diagnosis of completed P0 artifacts, no model selection."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from route_a.p0_baseline import Q, ALPHAS, METHODS, metric_arrays, sha, write


def calendar_bootstrap(daily,run):
    """Calendar-day blocks include no-service dates with zero numerator/count.

    The initial runner's union-of-issued-date blocks are retained as legacy
    observation-index diagnostics only; these calendar blocks govern reporting.
    """
    rng=np.random.default_rng(20261107);rows=[];grid_checks=[]
    for year in (2014,2012):
        farms=sorted({key[1] for key in daily if key[0]==year})
        union=np.unique(np.concatenate([daily[year,f,METHODS[0],.1]['origins'] for f in farms]))
        dates=np.arange(union.min(),union.max()+np.timedelta64(1,'D'),np.timedelta64(1,'D'))
        n=len(dates)
        grid_checks.append({'year':year,'calendar_days':n,'issued_union_days':len(union),'missing_all_farms_days':n-len(union)})
        draws=((rng.integers(0,n,size=(1000,int(np.ceil(n/14)),1))+np.arange(14))%n).reshape(1000,-1)[:,:n]
        for alpha in ALPHAS:
            estimates={};observed={}
            for method in METHODS:
                for metric in ('coverage','width','IS'):
                    vals=[];obs=[]
                    for farm in farms:
                        d=daily[year,farm,method,alpha];pos=np.searchsorted(dates,d['origins'])
                        numerator=np.zeros(n);denominator=np.zeros(n)
                        numerator[pos]=d[metric];denominator[pos]=d['n']
                        denom=denominator[draws].sum(axis=1)
                        assert (denom>0).all()
                        vals.append(numerator[draws].sum(axis=1)/denom)
                        obs.append(numerator.sum()/denominator.sum())
                    estimates[method,metric]=np.mean(vals,axis=0)
                    observed[method,metric]=float(np.mean(obs))
            for reference,method in [(METHODS[0],METHODS[1]),(METHODS[0],METHODS[2]),(METHODS[1],METHODS[2])]:
                for metric in ('coverage','width','IS'):
                    diff=estimates[method,metric]-estimates[reference,metric]
                    lo,hi=np.quantile(diff,[.025,.975])
                    rows.append(dict(year=year,confidence=1-alpha,method=method,reference=reference,metric=metric,
                        delta=observed[method,metric]-observed[reference,metric],ci_low=float(lo),ci_high=float(hi),
                        block_calendar_days=14,draws=1000))
    pd.DataFrame(rows).to_csv(run/'paired_calendar_blocks.csv',index=False)
    write(run/'calendar_bootstrap_protocol.json',dict(status='clarified_before_performance_analysis',
        reason='2012 has missing calendar days; union-of-issued-date blocks do not represent 14 calendar days',
        authoritative='paired_calendar_blocks.csv',superseded_for_reporting='paired_blocks.csv',
        raw_predictions_or_endpoints_changed=False,seed=20261107,draws=1000,calendar_checks=grid_checks,
        scope='conditional on fitted seed0 forests; no multiplicity correction; descriptive intervals'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=require_local_write(ROOT/args.run)
    assert json.loads((run/'completion.json').read_text())['status']=='complete'
    conditional=[];decomposition=[];cal_stats=[];distribution=[];point=[];daily={};weekday=[];missing_audit=[];transitions=[]
    for year,farms in ((2014,10),(2012,7)):
        for farm in range(1,farms+1):
            folder=run/f'gefcom{year}/farm_{farm:02d}'
            with np.load(folder/'predictions.npz') as z:
                y,history,pred,valid=z['y'],z['history'],z['pred'],z['valid']
            meta=pd.read_csv(folder/'samples.csv')
            calendar=pd.read_csv(folder/'calendar.csv')
            calendar['weekday']=pd.to_datetime(calendar.origin).dt.day_name()
            for (phase,day),g in calendar.groupby(['phase','weekday']):
                weekday.append(dict(year=year,farm=farm,phase=phase,weekday=day,calendar_days=len(g),
                    issued_days=int(g.input_valid.sum()),bad_history_days=int((~g.history_valid).sum()),
                    bad_nwp_days=int((~g.nwp_valid).sum()),bad_target_days=int((~g.target_valid).sum())))
            if year==2012:
                native=ROOT.parent/f'wind_datasets/nwp_benchmarks/gefcom2012_wind/data/windforecasts_wf{farm}.csv'
                w=pd.read_csv(native);w['issue']=pd.to_datetime(w.date.astype(str),format='%Y%m%d%H')
                batches={t:g.set_index('hors') for t,g in w.groupby('issue')}
                for phase in ('validation','evaluation'):
                    excluded=calendar[calendar.phase.eq(phase)&~calendar.nwp_valid]
                    minimums=[]
                    for now in pd.to_datetime(excluded.origin):
                        missing=[]
                        for age in (0,12,24):
                            batch=batches.get(now-pd.Timedelta(hours=age))
                            arr=np.full((24,2),np.nan) if batch is None else batch.reindex(np.arange(age+1,age+25))[['u','v']].to_numpy(float)
                            missing.append(int((~np.isfinite(arr).all(axis=1)).sum()))
                        minimums.append(min(missing))
                    assert not minimums or min(minimums)>0, 'Loader excluded a complete legal NWP batch'
                    missing_audit.append(dict(year=year,farm=farm,phase=phase,excluded_nwp_days=len(excluded),
                        candidate_ages_hours=[0,12,24],missing_target_hours_best_candidate_min=min(minimums) if minimums else None,
                        missing_target_hours_best_candidate_max=max(minimums) if minimums else None,
                        weekdays=excluded.weekday.value_counts().to_dict(),all_exclusions_verified_against_native_csv=True))
            with np.load(ROOT/f'out_gefcom_pooling_20261105/inputs/gefcom{year}/farm_{farm:02d}/features.npz') as z:
                canonical=z['GBM'].reshape(len(y),24,-1)
            # Canonical base features: history/stat/calendar 35, u24, v24, speed24.
            ws=canonical[:,0,83:107]
            np.testing.assert_allclose(ws,np.hypot(canonical[:,0,35:59],canonical[:,0,59:83]))
            train=meta.phase.eq('train').to_numpy()&valid
            q01,q99=np.quantile(ws[train],[.01,.99])
            for stage in ('validation','evaluation'):
                ids=np.flatnonzero(meta.phase.eq(stage));yy=y[ids]
                good=np.broadcast_to(valid[ids,None],yy.shape)
                delta=np.diff(yy,axis=1,prepend=history[ids,-1,None])
                windout=(ws[ids]<q01)|(ws[ids]>q99)
                events={}
                for name,mask in [('up',delta>.1),('down',delta<-.1)]:
                    events[name+'_cells']=int((mask&good).sum())
                    events[name+'_days']=int((mask&good).any(axis=1).sum())
                distribution.append(dict(year=year,farm=farm,stage=stage,
                    train_days=int(train.sum()),scored_days=int(valid[ids].sum()),
                    train_zero_fraction=float((y[train]==0).mean()),evaluation_zero_fraction=float((yy[good]==0).mean()),
                    train_near_zero_fraction=float((y[train]<=.01).mean()),evaluation_near_zero_fraction=float((yy[good]<=.01).mean()),
                    train_mean=float(y[train].mean()),evaluation_mean=float(yy[good].mean()),
                    wind_support_out_fraction=float(windout[good].mean()),**events))
                point.append(dict(year=year,farm=farm,stage=stage,
                    QRF_MAE=float(abs(yy-pred[ids,:,3])[good].mean()),
                    persistence_MAE=float(abs(yy-history[ids,-1,None])[good].mean()),
                    QRF_RMSE=float(np.sqrt(np.square(yy-pred[ids,:,3])[good].mean()))))
                with np.load(folder/f'{stage}_calibration.npz') as cal:
                    for alpha in ALPHAS:
                        qi=int(np.argmin(abs(Q-alpha/2)));qj=int(np.argmin(abs(Q-(1-alpha/2))))
                        for name in METHODS:
                            if name==METHODS[0]:
                                lo,hi=pred[ids,:,qi],pred[ids,:,qj]
                            else:
                                key=f'{name}_{1-alpha:.2f}'
                                lo,hi=cal[key+'_lo'],cal[key+'_hi']
                                corr=cal[key+'_correction']
                                cal_stats.append(dict(year=year,farm=farm,stage=stage,method=name,confidence=1-alpha,
                                    mean_signed_correction=float(corr[good].mean()),
                                    negative_correction_fraction=float((corr[good]<0).mean()),
                                    lower_physical_boundary_fraction=float((lo[good]==0).mean()),
                                    upper_physical_boundary_fraction=float((hi[good]==1).mean()),
                                    min_n=int(cal[key+'_n_cal'].min()),max_n=int(cal[key+'_n_cal'].max())))
                                if alpha==.1:
                                    near=good&(yy<=.01)
                                    raw_covered=(yy>=pred[ids,:,qi])&(yy<=pred[ids,:,qj])
                                    cal_covered=(yy>=lo)&(yy<=hi)
                                    lost=near&raw_covered&~cal_covered
                                    gained=near&~raw_covered&cal_covered
                                    transitions.append(dict(year=year,farm=farm,stage=stage,method=name,
                                        near_zero_cells=int(near.sum()),raw_covered_lost=int(lost.sum()),
                                        raw_missed_gained=int(gained.sum()),lost_with_negative_correction=int((lost&(corr<0)).sum())))
                            stats=metric_arrays(yy,lo,hi,alpha)
                            if stage=='evaluation':
                                daily[year,farm,name,alpha]=dict(origins=pd.to_datetime(meta.iloc[ids].origin).to_numpy(),
                                    n=good.sum(axis=1),**{k:np.where(good,v,0).sum(axis=1) for k,v in stats.items()})
                            for group,mask in [('all',np.ones(yy.shape,bool)),('up_ramp',delta>.1),('down_ramp',delta<-.1),('near_zero',yy<=.01)]:
                                m=good&mask
                                if m.any():
                                    lower=float((2/alpha*np.maximum(lo-yy,0))[m].mean())
                                    upper=float((2/alpha*np.maximum(yy-hi,0))[m].mean())
                                    width=float((hi-lo)[m].mean())
                                    decomposition.append(dict(year=year,farm=farm,stage=stage,method=name,confidence=1-alpha,
                                        group=group,cells=int(m.sum()),width=width,lower_penalty=lower,upper_penalty=upper,IS=width+lower+upper))
                            # These groups depend only on information available at issue time.
                            for group,mask in [('wind_in_train_1_99',~windout),('wind_out_train_1_99',windout),
                                ('last_power_le_.1',np.broadcast_to(history[ids,-1,None]<=.1,yy.shape)),
                                ('last_power_gt_.1',np.broadcast_to(history[ids,-1,None]>.1,yy.shape))]:
                                m=good&mask
                                if m.any():
                                    conditional.append(dict(year=year,farm=farm,stage=stage,method=name,confidence=1-alpha,
                                        group=group,cells=int(m.sum()),days=int(m.any(axis=1).sum()),
                                        **{k:float(v[m].mean()) for k,v in stats.items()}))
    for filename,data in [('conditional_inputs.csv',conditional),('interval_decomposition.csv',decomposition),
        ('calibration_diagnostics.csv',cal_stats),('distribution_shift.csv',distribution),('point_metrics.csv',point),
        ('availability_weekday.csv',weekday),('near_zero_transitions.csv',transitions)]:
        pd.DataFrame(data).to_csv(run/filename,index=False)
    write(run/'nwp_missing_audit.json',missing_audit)
    calendar_bootstrap(daily,run)
    write(run/'diagnosis_completion.json',{'status':'complete','source_sha256':sha(Path(__file__)),
        'scope':'descriptive retrospective subgroup analysis; no causal attribution of dataset versus model'})
    print('DIAGNOSIS_COMPLETE',run)


if __name__=='__main__':main()
