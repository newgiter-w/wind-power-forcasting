"""Versioned P1 allowlist publication, preserving prior public release files."""
from pathlib import Path
import argparse,json,hashlib,re
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from tools.build_p0_release import SOURCES as P0_SOURCES,safe

VERSION='0.3.0'
SOURCES=list(dict.fromkeys(P0_SOURCES+['docs/P1_METHODS.md','route_a/p1_protocol.json',
    'route_a/p1_distribution.py','route_a/p1_run.py','route_a/check_p1.py',
    'route_a/p1_analyze.py','route_a/p1_report.py','route_a/reviewer_verify_p1.py','tools/build_p1_release.py']))
RESULTS=['REPORT.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md',
    'metrics_macro.csv','metrics_per_farm.csv','scores_macro.csv','scores_per_farm.csv',
    'paired_calendar_blocks.csv','joint_gates.csv','selection.json','boundary_metrics.csv','state_support.csv',
    'forest_checks.json','input_manifest.json','code_manifest.json','completion.json','protocol.json',
    'environment.json','release_checks.json','verification_checks.json','analysis_completion.json',
    'literature_sources.json','evidence_values.json','evidence_precheck.json']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--source-commit',required=True);args=p.parse_args()
    run=require_local_write(ROOT/args.run);dest=require_local_write(ROOT/args.destination)
    assert dest.is_relative_to(ROOT/'artifacts') and (dest/'.git').is_dir()
    assert re.fullmatch(r'[0-9a-f]{40}',args.source_commit)
    for name in ('completion.json','analysis_completion.json'):
        assert json.loads((run/name).read_text(encoding='utf-8'))['status']=='complete'
    audit=json.loads((run/'EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['integrity_status'].lower() in ('pass','warn')
    prior=json.loads((dest/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    paths={e['path'] for e in prior['publication_files']}
    # Verify retained public history before any overwrite. Later reruns verify
    # the already-built manifest instead; all content remains allowlisted.
    for e in prior['publication_files']:
        assert hashlib.sha256((dest/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
    for source,rel in [(ROOT/f,f) for f in SOURCES]+[(run/f,f'docs/results/v{VERSION}/'+f) for f in RESULTS]:
        data=safe(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);paths.add(rel)
    for source in sorted((run/'review_trace').glob('*')):
        if source.is_file() and source.suffix in ('.py','.md','.json'):
            rel=f'docs/results/v{VERSION}/review_trace/'+source.name
            data=safe(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);paths.add(rel)
    entries=[]
    for rel in sorted(paths):
        assert not Path(rel).suffix.lower() in ('.npz','.npy','.joblib','.pkl','.parquet','.pt','.pth','.pdf','.env')
        data=(dest/rel).read_bytes();entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(version=VERSION,date='2026-11-07',route='A',scope='P1 retrospective fixed-QRF distribution and dual-tail recalibration',
        local_source_commit=args.source_commit,publication_files=entries,new_forests_fit=0,
        new_method_effectiveness_claims=False,joint_goal_met=False,outcome='no candidate satisfies the predeclared joint gate; retain CQR_signed84',
        assurance={'review_independence':'same-family','acceptance_status':'provisional'},
        excluded=['raw data','full leaf-response weights','hourly predictions','virtual environment','credentials','private Git history'])
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'version':VERSION,'files':len(entries)+1,'destination':str(dest)}))


if __name__=='__main__':main()
