from pathlib import Path
import json,hashlib
import pandas as pd
R=Path('D:/Study/wind_probability_route_a/experiments/route_a/20261107_p1_v030')
report=(R/'REPORT.md').read_text(encoding='utf-8')
m=pd.read_csv(R/'metrics_macro.csv');s=pd.read_csv(R/'scores_macro.csv');g=pd.read_csv(R/'joint_gates.csv');b=pd.read_csv(R/'paired_calendar_blocks.csv');bm=pd.read_csv(R/'boundary_metrics.csv');state=pd.read_csv(R/'state_support.csv')
mi=m.set_index(['year','stage','method','confidence','dimension','group']);si=s.set_index(['year','stage','method','group']);gi=g.set_index(['year','stage','method'])
expected=[]
for year in (2014,2012):
    for method in ['QRF_linear_raw','CQR_signed84','DualTail84','QRF_ECDF_raw','PIT84','PIT84_state','DualTail84_state']:
        a=mi.loc[year,'evaluation',method,.9,'regime','all'];up=mi.loc[year,'evaluation',method,.9,'regime','up_ramp'];zero=mi.loc[year,'evaluation',method,.9,'regime','near_zero'];ss=si.loc[year,'evaluation',method,'all'];gg=gi.loc[year,'evaluation',method]
        status='参考/原始对照' if not gg.eligible else ('通过' if gg.joint_pass else '未通过')
        expected.append(f'| {year} | {method} | {a.coverage:.2%} | {a.width:.6f} | {a.IS:.6f} | {ss.WIS:.6f} | {up.coverage:.2%} | {zero.coverage:.2%} | {status} |')
        expected.append(f'| {year} | {method} | {int(ss.nested_80_90)} | {int(ss.nested_90_95)} |')
    for method in ['DualTail84','PIT84','PIT84_state','DualTail84_state']:
        a=gi.loc[year,'evaluation',method];failed=[k[5:] for k in g.columns if k.startswith('pass_') and not a[k]]
        expected.append(f'| {method} | {(a.width_ratio-1)*100:+.2f}% | {(a.IS_ratio-1)*100:+.2f}% | {(a.WIS_ratio-1)*100:+.2f}% | {a.up_coverage_gain*100:+.2f} | {a.zero_coverage_gain*100:+.2f} | {", ".join(failed) or "无"} |')
        for group in ['all','up_ramp','near_zero']:
            row=b[(b.year==year)&(b.method==method)&(b.reference=='CQR_signed84')&(b.group==group)&(b.metric=='IS')].iloc[0]
            expected.append(f'| {year} | {method} | {group} | {row.delta:+.6f} | [{row.ci_low:+.6f}, {row.ci_high:+.6f}] |')
    for method in ['QRF_ECDF_raw','PIT84','PIT84_state']:
        ss=si.loc[year,'evaluation',method,'all'];a=bm[(bm.year==year)&(bm.stage=='evaluation')&(bm.method==method)].mean(numeric_only=True)
        expected.append(f'| {year} | {method} | {ss.CRPS:.6f} | {a.near_zero_Brier:.6f} | {a.mean_predicted_near_zero:.2%} | {a.observed_near_zero:.2%} | {a.mean_zero_mass:.2%} |')
    for label,part in state[(state.year==year)&(state.stage=='evaluation')].groupby('state'):
        expected.append(f'| {year} | {label} | {int(part.cells.sum())} | {part.local_fallback_fraction.mean():.2%} | {part.mean_lambda.mean():.3f} |')
for row in expected:assert row in report,row
a14=gi.loc[2014,'evaluation','DualTail84_state'];a12=gi.loc[2012,'evaluation','PIT84_state']
crps=1-si.loc[2012,'evaluation','PIT84_state','all'].CRPS/si.loc[2012,'evaluation','QRF_ECDF_raw','all'].CRPS
phrase=f'上爬坡/近零覆盖分别提高 {a14.up_coverage_gain*100:.2f}/{a14.zero_coverage_gain*100:.2f} 个百分点，但宽度增加 {(a14.width_ratio-1)*100:.2f}%，总体 IS/WIS 也未改善。2012 的条件 PIT 相对 CQR 改善 WIS {(1-a12.WIS_ratio)*100:.2f}%，相对原 ECDF 改善 CRPS {crps*100:.2f}%，但近零覆盖下降 {-a12.zero_coverage_gain*100:.2f} 个百分点。'
assert phrase in report
assert '[方法说明](../../../docs/P1_METHODS.md)' in report
assert (R/'../../../docs/P1_METHODS.md').resolve().is_file()
out={'status':'pass','table_rows_checked':len(expected),'headline_numeric_claim_checked':True,'report_sha256':hashlib.sha256((R/'REPORT.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(R/'review_trace/p1_report_verify.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
