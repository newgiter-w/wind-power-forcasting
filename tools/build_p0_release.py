"""Copy an explicit, audited P0 publication allowlist into a publication clone.

Never copies datasets, model objects, per-issue predictions or inherited Git history.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write

VERSION='0.2.0'
SOURCES=['README.md','AGENTS.md','CHANGELOG.md','.gitignore','requirements-route-a.txt',
    'docs/ROUTE_A_PLAN.md','docs/ISOLATION.md','docs/PUBLISHING.md','docs/COMPUTE_ENV.md',
    'docs/environment.lock.txt','docs/verification.json','docs/snapshot_summary.json','docs/literature/references.bib',
    'route_a/__init__.py','route_a/paths.py','route_a/findings.md','route_a/p0_protocol.json','route_a/p0_baseline.py',
    'route_a/p0_diagnose.py','route_a/p0_report.py','route_a/p0_verify.py','route_a/reviewer_verify_p0.py','route_a/check_p0.py',
    'tools/run_route_a.py','tools/verify_isolation.py','tools/runtime_guard/sitecustomize.py',
    'tools/build_p0_release.py','qrf_witness.py',
    'gefcom_protocol_v2.py','gefcom2014_experiment.py','run_gefcom_tail_dynamic.py',
    'strong_baselines.py','run_p2_2012.py','online_calibration.py']
RESULTS=['REPORT.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md',
    'metrics_macro.csv','metrics_per_farm.csv','scores_macro.csv','scores_per_farm.csv','pinball_per_farm.csv',
    'data_quality.csv','data_support.csv','point_metrics.csv','distribution_shift.csv',
    'conditional_inputs.csv','interval_decomposition.csv','calibration_diagnostics.csv',
    'availability_weekday.csv','nwp_missing_audit.json','near_zero_transitions.csv',
    'paired_calendar_blocks.csv','calendar_bootstrap_protocol.json','protocol_deviations.json','protocol.json','environment.json',
    'reproduction.json','input_manifest.json','code_manifest.json','completion.json','diagnosis_completion.json',
    'evidence_precheck.json','evidence_values.json','verification_checks.json','release_checks.json']


def safe(p):
    b=Path(p).read_bytes()
    for pattern in (rb'gh[pousr]_[A-Za-z0-9]{20,}',rb'github_pat_[A-Za-z0-9_]{20,}',rb'-----BEGIN [^-]*PRIVATE KEY-----'):
        if re.search(pattern,b):raise ValueError('Potential credential material in '+str(p))
    return b


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--source-commit',required=True)
    args=p.parse_args();run=require_local_write(ROOT/args.run);dest=require_local_write(ROOT/args.destination)
    assert re.fullmatch(r'[0-9a-f]{40}',args.source_commit)
    assert dest.is_relative_to(ROOT/'artifacts') and (dest/'.git').is_dir()
    assert json.loads((run/'completion.json').read_text(encoding='utf-8'))['status']=='complete'
    assert json.loads((run/'diagnosis_completion.json').read_text(encoding='utf-8'))['status']=='complete'
    # A FAIL must be corrected and re-reviewed before a release is published.
    audit=json.loads((run/'EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['integrity_status'].lower() in ('pass','warn')
    entries=[]
    for source,rel in [(ROOT/f,f) for f in SOURCES]+[(run/f,'docs/results/v'+VERSION+'/'+f) for f in RESULTS]:
        b=safe(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
        entries.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    # Include fresh review traces without exposing local runtime/config contents.
    trace=run/'review_trace'
    for source in sorted(trace.glob('*')):
        if source.is_file() and source.suffix in ('.json','.md','.py'):
            rel='docs/results/v'+VERSION+'/review_trace/'+source.name
            b=safe(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
            entries.append(dict(path=rel,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    retained='archives/route-a-v0.1.1-legacy-python.zip'
    previous=json.loads((dest/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    expected=next(e['sha256'] for e in previous['publication_files'] if e['path']==retained)
    b=(dest/retained).read_bytes()
    assert hashlib.sha256(b).hexdigest()==expected
    entries.append(dict(path=retained,bytes=len(b),sha256=expected,retained_from='route-a-v0.1.1'))
    manifest=dict(version=VERSION,date='2026-11-07',route='A',scope='P0 retrospective baseline reproduction and diagnosis',
        local_source_commit=args.source_commit,execution_code_manifest='docs/results/v0.2.0/code_manifest.json',
        new_method_effectiveness_claims=False,publication_files=entries,
        excluded=['raw data','hourly predictions','model weights','virtual environment','private authentication','original project Git history'],
        assurance={'review_independence':'same-family','acceptance_status':'provisional'})
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'destination':str(dest),'files':len(entries)+1,'version':VERSION}))


if __name__=='__main__':main()
