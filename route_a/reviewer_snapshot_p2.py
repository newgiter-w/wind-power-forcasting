"""Read-only final metadata/continuity verification for P2 fresh review."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/route_a/20261107_p2_v040'
P0=ROOT/'experiments/route_a/20261107_p0_v020'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
old={Path(e['path']).as_posix():e for e in read(P0/'input_manifest.json')}
forest=read(RUN/'forest_hash_continuity.json')
for entry in forest:
    path=f"out_gefcom_revision_20261104/qrf/gefcom{entry['year']}/farm_{entry['farm']:02d}/leaf_10/seed_0/model.joblib"
    assert sha(ROOT/path)==old[path]['sha256']==entry['QRF_sha256']
for entry in read(RUN/'evaluation_code_manifest.json'):assert sha(ROOT/entry['path'])==entry['sha256']
assert sha(ROOT/'route_a/p2_analyze.py')==read(RUN/'analysis_completion.json')['source_sha256']
assert read(RUN/'protocol.json')==read(ROOT/'route_a/p2_protocol.json')
cal=read(RUN/'calibration_checks.json');assert len(cal)==170
assert all(x['raw_nontrain_crossings']==0 for x in cal)
env=read(RUN/'environment_witness.json')
assert env['followup_addendum']['result']=='pass'
assert env['followup_addendum']['remaining_doc_vs_reality_divergences']==[]
numeric=read(RUN/'review_trace/independent_numeric_verification.json')
paths=[ROOT/p for p in ['AGENTS.md','route_a/p2_protocol.json','route_a/p2_models.py','route_a/p2_train.py','route_a/check_p2.py','route_a/p2_evaluate.py','route_a/p2_analyze.py','route_a/p2_report.py','route_a/p0_baseline.py','gefcom_protocol_v2.py','gefcom2014_experiment.py','run_gefcom_tail_dynamic.py','strong_baselines.py','run_p2_2012.py','docs/P2_METHODS.md','docs/P2_COMPUTE_ENV.md','docs/P2_COMPUTE_SPEC.json','docs/environment-temporal.lock.txt','requirements-route-a.txt','requirements-route-a-temporal.txt','route_a/reviewer_verify_p2.py']]
paths += [RUN/p for p in ['REPORT.md','protocol.json','training_completion.json','completion.json','analysis_completion.json','training_summary.csv','training_diagnostics.csv','learning_curves.csv','data_contracts.json','input_manifest.json','training_code_manifest.json','evaluation_code_manifest.json','calibration_checks.json','forest_hash_continuity.json','release_checks.json','environment.json','metrics_per_farm.csv','metrics_seed_macro.csv','metrics_macro.csv','scores_per_farm.csv','scores_seed_macro.csv','scores_macro.csv','paired_calendar_blocks.csv','joint_gates.csv','joint_gates_by_seed.csv','selection.json','boundary_diagnostics.csv','environment_witness.md','environment_witness.json','evidence_precheck.json','evidence_values.json','review_trace/independent_numeric_verification.json','review_trace/training_subreview/training_subreview.json','review_trace/training_subreview/verify_training.py']]
paths+=[P0/'input_manifest.json']
for year in [2014,2012]:
    folder=RUN/f'gefcom{year}/farm_01';paths+=[folder/p for p in ['preprocessing.joblib','indices.npz','data_contract.json']]
    for family in ['MLP','TCN','GRU']:
        sf=folder/family/'seed_0';paths+=[sf/p for p in ['fit.json','selection_history.csv','refit_history.csv','model.pt','heads.npz','validation.npz','evaluation.npz']]
record={'status':'pass','forest_hashes_checked':len(forest),'evaluation_source_hashes_checked':2,'analysis_source_hash_matches':True,'protocol_copies_equal':True,'calibration_witnesses':len(cal),'independent_cqr_lead_ranks':sum(r['lead_ranks'] for r in numeric['checks']['independent_cqr']),'audited_input_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in paths}}
(RUN/'review_trace/reviewer_snapshot.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in record.items() if k!='audited_input_hashes'},indent=2))
