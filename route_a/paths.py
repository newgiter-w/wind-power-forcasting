from pathlib import Path
from datetime import datetime,timezone,timedelta
import os,re
PROJECT_ROOT=Path(__file__).resolve().parents[1]
ORIGINAL_ROOT=Path('D:/Study/wind_retrieval_interval').resolve()
DATA_ROOT=Path(os.environ.get('WIND_DATA_ROOT','D:/Study/wind_datasets')).resolve()
OUTPUT_ROOT=PROJECT_ROOT/'experiments'/'route_a'
assert PROJECT_ROOT!=ORIGINAL_ROOT and not PROJECT_ROOT.is_relative_to(ORIGINAL_ROOT)
def new_run_dir(label='experiment'):
    if not re.fullmatch(r'[A-Za-z0-9_-]+',label):raise ValueError('Use a simple run label')
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    p=OUTPUT_ROOT/(stamp+'_'+label);p.mkdir(parents=True,exist_ok=False);return p
def require_local_write(path):
    p=Path(path).resolve()
    if not p.is_relative_to(PROJECT_ROOT):raise PermissionError('Route-A write outside project: '+str(p))
    return p
