"""Independent, read-only P0 numerical audit; writes only review-trace evidence."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np
import pandas as pd
from gefcom_protocol_v2 import prepare

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'experiments/route_a/20261107_p0_v020'
OUT = RUN / 'review_trace'
OUT.mkdir(exist_ok=True)
def read(name):
    return pd.read_csv(RUN / name, float_precision='round_trip')
def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

report = {'review_script_sha256': digest(Path(__file__)), 'training_run': False}
manifests = {}
for name in ('input_manifest.json', 'code_manifest.json'):
    entries = json.loads((RUN / name).read_text())
    bad = []
    for entry in entries:
        path = Path(entry['path'])
        if not path.is_absolute(): path = ROOT / path
        if not path.is_file() or digest(path) != entry['sha256']:
            bad.append(str(path))
    manifests[name] = {'entries': len(entries), 'mismatches': bad}
report['manifests'] = manifests
assert all(not m['mismatches'] for m in manifests.values())
report['diagnosis_source_hash_matches'] = digest(ROOT/'route_a/p0_diagnose.py') == json.loads((RUN/'diagnosis_completion.json').read_text())['source_sha256']
assert report['diagnosis_source_hash_matches']

metrics, summaries, pins = read('metrics_per_farm.csv'), read('scores_per_farm.csv'), read('pinball_per_farm.csv')
pt, transitions, quality = read('point_metrics.csv'), read('near_zero_transitions.csv'), read('data_quality.csv')
decomp, conditional, calibration = read('interval_decomposition.csv'), read('conditional_inputs.csv'), read('calibration_diagnostics.csv')
support, distribution, weekday = read('data_support.csv'), read('distribution_shift.csv'), read('availability_weekday.csv')
errors = {}; counts = {}; daily = {}; macro_rows = []; totals = {}; selected = []
def check(kind, actual, expected):
    error = float(np.max(np.abs(np.asarray(actual, dtype=float)-np.asarray(expected, dtype=float))))
    errors[kind] = max(errors.get(kind, 0.), error)
    counts[kind] = counts.get(kind, 0)+1
    assert error < 1e-11, (kind, actual, expected, error)

for year, nfarms in ((2014,10),(2012,7)):
    raw_farms, _ = prepare(year)
    totals[year] = {'issued_days':0,'scored_days':0,'calendar_days':0,'nested80_90':0,'nested90_95':0,'lost_near_zero':0,'gained_near_zero':0,'negative_lost':0}
    for farm in range(1,nfarms+1):
        folder = RUN / f'gefcom{year}/farm_{farm:02d}'
        with np.load(folder/'predictions.npz') as z:
            y, h, pred, good, origins, q = [z[k] for k in ('y','history','pred','valid','origins','quantiles')]
        samples = pd.read_csv(folder/'samples.csv')
        calendar = pd.read_csv(folder/'calendar.csv')
        np.testing.assert_equal(raw_farms[farm]['y'],y)
        wind = np.hypot(raw_farms[farm]['nwp'][:,:,-2],raw_farms[farm]['nwp'][:,:,-1])
        train = (samples.phase=='train').to_numpy()&good
        wlo,whi = np.quantile(wind[train],[.01,.99])
        calendar['weekday'] = pd.to_datetime(calendar.origin).dt.day_name()
        for (phase,day),cg in calendar.groupby(['phase','weekday']):
            wr=weekday[(weekday.year==year)&(weekday.farm==farm)&(weekday.phase==phase)&(weekday.weekday==day)].iloc[0]
            check('weekday_availability',[len(cg),cg.input_valid.sum(),(~cg.history_valid).sum(),(~cg.nwp_valid).sum(),(~cg.target_valid).sum()],wr[['calendar_days','issued_days','bad_history_days','bad_nwp_days','bad_target_days']])
        for phase, cg in calendar.groupby('phase'):
            row = quality[(quality.year==year)&(quality.farm==farm)&(quality.phase==phase)].iloc[0]
            issued = (samples.phase==phase).to_numpy()
            check('data_quality', [row.calendar_days,row.issued_days,row.valid_label_days,row.input_service_rate,row.bad_history_days,row.bad_nwp_days], [len(cg),issued.sum(),good[issued].sum(),issued.sum()/len(cg),(~cg.history_valid).sum(),(~cg.nwp_valid).sum()])
            if phase=='evaluation':
                totals[year]['calendar_days'] += len(cg); totals[year]['issued_days'] += int(issued.sum()); totals[year]['scored_days'] += int(good[issued].sum())
        for stage in ('validation','evaluation'):
            ids = np.flatnonzero((samples.phase==stage).to_numpy())
            yy, pp = y[ids], pred[ids]
            mask = np.repeat(good[ids,None],24,axis=1)
            change = np.concatenate([yy[:,:1]-h[ids,-1,None],np.diff(yy,axis=1)],axis=1)
            group_masks = {'all':np.ones(yy.shape,bool),'up_ramp':change>.1,'down_ramp':change<-.1,'near_zero':yy<=.01}
            windout=(wind[ids]<wlo)|(wind[ids]>whi)
            su=support[(support.year==year)&(support.farm==farm)&(support.stage==stage)].iloc[0]
            check('data_support',[wlo,whi,windout[mask].mean(),(yy[mask]==0).mean(),(yy[mask]<=.01).mean(),yy[mask].std(),y[train].std(),samples.iloc[ids].nwp_age_hours.max()],su[['wind_train_p01','wind_train_p99','wind_outside_train_p01_p99_fraction','target_zero_fraction','target_near_zero_fraction','target_std','train_target_std','nwp_age_max_hours']])
            ds=distribution[(distribution.year==year)&(distribution.farm==farm)&(distribution.stage==stage)].iloc[0]
            check('distribution_shift',[train.sum(),good[ids].sum(),(y[train]==0).mean(),(yy[mask]==0).mean(),(y[train]<=.01).mean(),(yy[mask]<=.01).mean(),y[train].mean(),yy[mask].mean(),windout[mask].mean(),(group_masks['up_ramp']&mask).sum(),(group_masks['up_ramp']&mask).any(axis=1).sum(),(group_masks['down_ramp']&mask).sum(),(group_masks['down_ramp']&mask).any(axis=1).sum()],ds[['train_days','scored_days','train_zero_fraction','evaluation_zero_fraction','train_near_zero_fraction','evaluation_near_zero_fraction','train_mean','evaluation_mean','wind_support_out_fraction','up_cells','up_days','down_cells','down_days']])
            for qi, quantile in enumerate(q):
                loss = np.where(yy>=pp[:,:,qi],quantile*(yy-pp[:,:,qi]),(1-quantile)*(pp[:,:,qi]-yy))
                row = pins[(pins.year==year)&(pins.farm==farm)&(pins.stage==stage)&(pins['quantile']==quantile)].iloc[0]
                check('pinball',loss[mask].mean(),row.pinball)
            pointrow = pt[(pt.year==year)&(pt.farm==farm)&(pt.stage==stage)].iloc[0]
            mae = abs(yy-pp[:,:,3])[mask].mean(); persistence = abs(yy-h[ids,-1,None])[mask].mean()
            check('point_metrics',[mae,persistence,np.sqrt(np.square(yy-pp[:,:,3])[mask].mean())],pointrow[['QRF_MAE','persistence_MAE','QRF_RMSE']])
            with np.load(folder/f'{stage}_calibration.npz') as cal:
                assert np.array_equal(cal['indices'],ids)
                for method in ('QRF_raw','QRF_CQR_static','QRF_CQR_rolling84'):
                    interval_list=[]; wis = .5*abs(yy-pp[:,:,3]); daycov=[]
                    for alpha in (.2,.1,.05):
                        qi=np.flatnonzero(np.isclose(q,alpha/2))[0]; qj=np.flatnonzero(np.isclose(q,1-alpha/2))[0]
                        if method=='QRF_raw': lo,hi=pp[:,:,qi],pp[:,:,qj]
                        else:
                            key=f'{method}_{1-alpha:.2f}';lo,hi=cal[key+'_lo'],cal[key+'_hi']
                            assert (cal[key+'_max_label_end']<=origins[ids]).all()
                            cr=calibration[(calibration.year==year)&(calibration.farm==farm)&(calibration.stage==stage)&(calibration.method==method)&(calibration.confidence==1-alpha)].iloc[0]
                            correction=cal[key+'_correction']
                            check('calibration_diagnostics',[correction[mask].mean(),(correction[mask]<0).mean(),(lo[mask]==0).mean(),(hi[mask]==1).mean(),cal[key+'_n_cal'].min(),cal[key+'_n_cal'].max()],cr[['mean_signed_correction','negative_correction_fraction','lower_physical_boundary_fraction','upper_physical_boundary_fraction','min_n','max_n']])
                        interval_list.append((lo,hi))
                        score=hi-lo + np.where(yy<lo,2*(lo-yy)/alpha,0) + np.where(yy>hi,2*(yy-hi)/alpha,0)
                        arrays={'coverage':((lo<=yy)&(yy<=hi)).astype(float),'width':hi-lo,'IS':score,'lower_miss':(yy<lo).astype(float),'upper_miss':(yy>hi).astype(float)}
                        wis += alpha*score/2
                        for cg,cm in [('wind_in_train_1_99',~windout),('wind_out_train_1_99',windout),('last_power_le_.1',np.broadcast_to(h[ids,-1,None]<=.1,yy.shape)),('last_power_gt_.1',np.broadcast_to(h[ids,-1,None]>.1,yy.shape))]:
                            keep=mask&cm
                            if not keep.any(): continue
                            ci=conditional[(conditional.year==year)&(conditional.farm==farm)&(conditional.stage==stage)&(conditional.method==method)&(conditional.confidence==1-alpha)&(conditional.group==cg)].iloc[0]
                            check('conditional_inputs',[keep.sum(),keep.any(axis=1).sum(),*[v[keep].mean() for v in arrays.values()]],ci[['cells','days',*arrays]])
                        daycov.append(float(((lo<=yy)&(yy<=hi)).all(axis=1)[good[ids]].mean()))
                        for group,gm in group_masks.items():
                            keep=mask&gm
                            if not keep.any(): continue
                            row = metrics[(metrics.year==year)&(metrics.farm==farm)&(metrics.stage==stage)&(metrics.method==method)&(metrics.confidence==1-alpha)&(metrics.dimension=='regime')&(metrics.group==group)].iloc[0]
                            check('interval_metrics',[v[keep].mean() for v in arrays.values()],row[list(arrays)])
                            check('metric_counts',[keep.sum(),keep.any(axis=1).sum()],row[['cells','days']])
                            dr=decomp[(decomp.year==year)&(decomp.farm==farm)&(decomp.stage==stage)&(decomp.method==method)&(decomp.confidence==1-alpha)&(decomp.group==group)].iloc[0]
                            check('interval_decomposition', [np.maximum(lo-yy,0)[keep].mean()*2/alpha,np.maximum(yy-hi,0)[keep].mean()*2/alpha,score[keep].mean()],dr[['lower_penalty','upper_penalty','IS']])
                        if stage=='evaluation':
                            daily[year,farm,method,alpha] = (origins[ids],mask.sum(axis=1),{k:np.where(mask,v,0).sum(axis=1) for k,v in arrays.items()})
                        if method!='QRF_raw' and alpha==.1:
                            raw_hit=(pp[:,:,qi]<=yy)&(yy<=pp[:,:,qj]);hit=(lo<=yy)&(yy<=hi);near=mask&(yy<=.01)
                            lost=near&raw_hit&~hit; gained=near&~raw_hit&hit
                            tr=transitions[(transitions.year==year)&(transitions.farm==farm)&(transitions.stage==stage)&(transitions.method==method)].iloc[0]
                            values=[near.sum(),lost.sum(),gained.sum(),(lost&(cal[key+'_correction']<0)).sum()]
                            check('near_zero_transitions',values,tr[['near_zero_cells','raw_covered_lost','raw_missed_gained','lost_with_negative_correction']])
                            if stage=='evaluation' and method=='QRF_CQR_rolling84':
                                for k,v in zip(('lost_near_zero','gained_near_zero','negative_lost'),values[1:]):totals[year][k]+=int(v)
                    sr=summaries[(summaries.year==year)&(summaries.farm==farm)&(summaries.stage==stage)&(summaries.method==method)].iloc[0]
                    violations=[int((((a[0]<b[0])|(a[1]>b[1]))&mask).sum()) for a,b in zip(interval_list,interval_list[1:])]
                    check('scores',[(wis/3.5)[mask].mean(),mae,persistence,*daycov,*violations],sr[['WIS','median_MAE','persistence_MAE','day_coverage_0.80','day_coverage_0.90','day_coverage_0.95','nested_80_in_90_violations','nested_90_in_95_violations']])
                    if stage=='evaluation' and method=='QRF_CQR_rolling84':
                        totals[year]['nested80_90']+=violations[0];totals[year]['nested90_95']+=violations[1]

macro=read('metrics_macro.csv').sort_values(['year','stage','method','confidence','dimension','group']).reset_index(drop=True)
mm=metrics.groupby(['year','stage','method','confidence','dimension','group'])[['coverage','width','IS','lower_miss','upper_miss']].mean().reset_index()
check('macro_metrics_all_rows',mm.iloc[:,-5:].to_numpy(),macro.iloc[:,-5:].to_numpy())
sm=read('scores_macro.csv').sort_values(['year','stage','method']).reset_index(drop=True)
computed=summaries.groupby(['year','stage','method']).mean(numeric_only=True).drop(columns='farm').reset_index()
check('macro_scores_all_rows',computed.iloc[:,3:].to_numpy(),sm.iloc[:,3:].to_numpy())

rng=np.random.default_rng(20261107);blocks=read('paired_calendar_blocks.csv');block_values=[]
for year,nfarms in ((2014,10),(2012,7)):
    # Full evaluation calendar from the saved source calendar, including unserved dates.
    cal=pd.read_csv(RUN/f'gefcom{year}/farm_01/calendar.csv')
    dates=pd.to_datetime(cal.loc[cal.phase=='evaluation','origin']).to_numpy()
    n=len(dates);assert (np.diff(dates)==np.timedelta64(1,'D')).all()
    starts=rng.integers(0,n,size=(1000,math.ceil(n/14)))
    draws=(starts[:,:,None]+np.arange(14))%n;draws=draws.reshape(1000,-1)[:,:n]
    for alpha in (.2,.1,.05):
        obs={};boot={}
        for method in ('QRF_raw','QRF_CQR_static','QRF_CQR_rolling84'):
            for metric in ('coverage','width','IS'):
                fm=[];fs=[]
                for farm in range(1,nfarms+1):
                    times,count,values=daily[year,farm,method,alpha]
                    positions=np.searchsorted(dates,times); numerator=np.zeros(n); denominator=np.zeros(n)
                    numerator[positions]=values[metric];denominator[positions]=count
                    fm.append(numerator.sum()/denominator.sum());fs.append(numerator[draws].sum(axis=1)/denominator[draws].sum(axis=1))
                obs[method,metric]=np.mean(fm);boot[method,metric]=np.mean(fs,axis=0)
        for reference,method in [('QRF_raw','QRF_CQR_static'),('QRF_raw','QRF_CQR_rolling84'),('QRF_CQR_static','QRF_CQR_rolling84')]:
            for metric in ('coverage','width','IS'):
                row=blocks[(blocks.year==year)&(blocks.confidence==1-alpha)&(blocks.reference==reference)&(blocks.method==method)&(blocks.metric==metric)].iloc[0]
                values=[obs[method,metric]-obs[reference,metric],*np.quantile(boot[method,metric]-boot[reference,metric],[.025,.975])]
                check('calendar_bootstrap',values,row[['delta','ci_low','ci_high']])
                if alpha==.1 and metric=='IS' and method=='QRF_CQR_rolling84': block_values.append({'year':year,'reference':reference,'delta':float(values[0]),'ci_low':float(values[1]),'ci_high':float(values[2])})

ev=json.loads((RUN/'evidence_values.json').read_text())['rows'][0]
with (RUN/'metrics_macro.csv').open(newline='',encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
    match=[r for r in rows if all(r[k]==ev[k] for k in ('year','stage','method','confidence','dimension','group'))]
    assert len(match)==1 and match[0]==ev
report.update(max_abs_errors=errors,checked_comparisons=counts,evaluation_totals=totals,calendar_bootstrap_90_IS=block_values,evidence_values_exact_csv_match=True)
report['audited_artifact_hashes']={p.name:digest(p) for p in RUN.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.md')}
(OUT/'independent_numerical_verification.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='audited_artifact_hashes'},indent=2))
