from pathlib import Path
import os,json,sys
ROOT=Path(__file__).resolve().parents[1]
assert os.environ.get('ROUTE_A_GUARD')=='1'
assert Path.cwd().resolve()==ROOT
local=ROOT/'.runtime_cache'/'guard_probe.txt';local.parent.mkdir(exist_ok=True)
local.write_text('route-a',encoding='utf-8');assert local.read_text()=='route-a'
# Probe a harmless non-project path; the guard must reject before creating it.
outside=ROOT.parent/'project_backups'/'route_a_guard_probe_should_not_exist.txt'
assert not outside.exists()
blocked=False
try:outside.write_text('must not be written',encoding='utf-8')
except PermissionError:blocked=True
assert blocked and not outside.exists()
try:(ROOT.parent/'wind_retrieval_interval'/'config.py').read_text(encoding='utf-8')
except Exception as e:raise AssertionError('Read-only reference access failed') from e
import sitecustomize
for target in [ROOT.parent/'wind_retrieval_interval'/'x.txt',ROOT.parent/'wind_datasets'/'x.txt']:
    try:sitecustomize._check(target)
    except PermissionError:pass
    else:raise AssertionError('Protected path accepted')
print(json.dumps({'inside_write':True,'outside_write_blocked':True,'source_read_allowed':True,'source_and_dataset_writes_rejected':True,'guard_loaded_from':sitecustomize.__file__},ensure_ascii=False))
