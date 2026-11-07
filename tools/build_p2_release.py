"""Explicit P2 publication allowlist; retains prior public versions verbatim."""
from pathlib import Path
import argparse,json,hashlib,re
from route_a.paths import PROJECT_ROOT as ROOT,require_local_write
from tools.build_p1_release import SOURCES as PRIOR_SOURCES
from tools.build_p0_release import safe

VERSION='0.4.0'
SOURCES=list(dict.fromkeys(PRIOR_SOURCES+['requirements-route-a-temporal.txt','docs/P2_COMPUTE_ENV.md','docs/P2_METHODS.md',
    'docs/P2_COMPUTE_SPEC.json','docs/environment-temporal.lock.txt','route_a/p2_protocol.json',
    'route_a/p2_models.py','route_a/check_p2.py','route_a/p2_train.py','route_a/p2_evaluate.py',
    'route_a/p2_analyze.py','route_a/p2_report.py','route_a/reviewer_verify_p2.py','route_a/reviewer_snapshot_p2.py','route_a/reviewer_finalize_p2.py','tools/build_p2_release.py']))
RESULTS=['REPORT.md','EXPERIMENT_AUDIT.md','EXPERIMENT_AUDIT.json','CLAIMS_FROM_RESULTS.md',
    'metrics_per_farm.csv','metrics_seed_macro.csv','metrics_macro.csv',
    'scores_per_farm.csv','scores_seed_macro.csv','scores_macro.csv','boundary_diagnostics.csv',
    'paired_calendar_blocks.csv','joint_gates.csv','joint_gates_by_seed.csv','selection.json',
    'training_summary.csv','training_diagnostics.csv','learning_curves.csv','data_contracts.json',
    'calibration_checks.json','forest_hash_continuity.json','input_manifest.json',
    'training_code_manifest.json','evaluation_code_manifest.json','protocol.json','environment.json',
    'training_completion.json','completion.json','analysis_completion.json','release_checks.json',
    'environment_witness.md','environment_witness.json','evidence_values.json','evidence_precheck.json']


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--destination',required=True)
    p.add_argument('--source-commit',required=True);args=p.parse_args()
    run=require_local_write(ROOT/args.run);dest=require_local_write(ROOT/args.destination)
    assert dest.is_relative_to(ROOT/'artifacts') and (dest/'.git').is_dir()
    assert re.fullmatch(r'[0-9a-f]{40}',args.source_commit)
    for name in ('training_completion.json','completion.json','analysis_completion.json'):
        assert json.loads((run/name).read_text(encoding='utf-8'))['status']=='complete'
    audit=json.loads((run/'EXPERIMENT_AUDIT.json').read_text(encoding='utf-8'))
    assert audit['integrity_status'].lower() in ('pass','warn')
    previous=json.loads((dest/'RELEASE_MANIFEST.json').read_text(encoding='utf-8'))
    paths={e['path'] for e in previous['publication_files']}
    for e in previous['publication_files']:
        assert hashlib.sha256((dest/e['path']).read_bytes()).hexdigest()==e['sha256'],e['path']
    for source,rel in [(ROOT/f,f) for f in SOURCES]+[(run/f,f'docs/results/v{VERSION}/'+f) for f in RESULTS]:
        data=safe(source);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);paths.add(rel)
    trace=run/'review_trace'
    for source in sorted(trace.rglob('*')):
        if source.is_file() and source.suffix in ('.py','.md','.json'):
            rel=f'docs/results/v{VERSION}/review_trace/'+source.relative_to(trace).as_posix()
            target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(safe(source));paths.add(rel)
    entries=[]
    for rel in sorted(paths):
        assert not Path(rel).suffix.lower() in ('.npz','.npy','.joblib','.pkl','.parquet','.pt','.pth','.pdf','.env')
        data=(dest/rel).read_bytes();entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    manifest=dict(version=VERSION,date='2026-11-07',route='A',scope='P2 same-information noncrossing MLP/TCN/GRU, same signed84 CQR',
        local_source_commit=args.source_commit,publication_files=entries,neural_final_models=153,neural_seeds=[0,1,2],QRF_refits=0,
        frozen_selection=selection,retrospective=True,submission_or_general_superiority_acceptance=False,
        assurance={'review_independence':'same-family','acceptance_status':'provisional'},
        excluded=['raw data','hourly predictions','network/forest weights','preprocessor objects','virtual environment','credentials','private Git history'])
    (dest/'RELEASE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'version':VERSION,'files':len(entries)+1,'destination':str(dest)}))


if __name__=='__main__':main()
