"""Attach checked snapshot to reviewer-written verdict without changing experiments."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
TRACE=ROOT/'experiments/route_a/20261107_p2_v040/review_trace'
path=TRACE/'001-review.verdict.json'
verdict=json.loads(path.read_text(encoding='utf-8'))
snapshot=json.loads((TRACE/'reviewer_snapshot.json').read_text(encoding='utf-8'))
assert snapshot['status']=='pass'
for relative,digest in snapshot['audited_input_hashes'].items():
    with (ROOT/relative).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest,relative
verdict['audited_input_hashes']=snapshot['audited_input_hashes']
verdict['generated_at']=datetime.now(timezone.utc).isoformat()
verdict['raw_response_sha256']=hashlib.sha256((TRACE/'001-review.response.md').read_bytes()).hexdigest()
verdict['numerical_verification']=json.loads((TRACE/'independent_numeric_verification.json').read_text(encoding='utf-8'))
training=json.loads((TRACE/'training_subreview/training_subreview.json').read_text(encoding='utf-8'))
verdict['training_verification_summary']=training['all_model_summary']
path.write_text(json.dumps(verdict,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
print(json.dumps({'verdict':verdict['verdict'],'claims':{x['id']:x['claim_supported'] for x in verdict['claims']},'audited_hashes':len(verdict['audited_input_hashes']),'raw_response':str(TRACE/'001-review.response.md'),'structured_verdict':str(path)},indent=2))
