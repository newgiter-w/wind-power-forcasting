from pathlib import Path
import argparse,os,sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=Path('D:/Study/wind_retrieval_interval').resolve()
def main():
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--script');args,rest=p.parse_known_args()
    assert ROOT!=ORIGINAL and not ROOT.is_relative_to(ORIGINAL)
    assert (ROOT/'.git').is_dir() and not (ROOT/'.git').is_symlink()
    if args.check:
        print(json.dumps({'route':'A','project':str(ROOT),'original_read_only':str(ORIGINAL),'outputs':str(ROOT/'experiments/route_a'),'guard':'Python audit hook, not an OS sandbox'},ensure_ascii=False,indent=2));return
    if not args.script:p.error('--script or --check required')
    script=(ROOT/args.script).resolve()
    if not script.is_relative_to(ROOT) or not script.is_file():p.error('Script must exist inside isolated project')
    cache=ROOT/'.runtime_cache';tmp=cache/'tmp';tmp.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(ROUTE_A_GUARD='1',ROUTE_A_PROJECT_ROOT=str(ROOT),TEMP=str(tmp),TMP=str(tmp),TMPDIR=str(tmp),MPLCONFIGDIR=str(cache/'matplotlib'),TORCH_HOME=str(cache/'torch'),XDG_CACHE_HOME=str(cache),PYTHONIOENCODING='utf-8')
    env['PYTHONPATH']=os.pathsep.join([str(ROOT/'tools/runtime_guard'),str(ROOT)])
    result=subprocess.run([sys.executable,'-u',str(script),*rest],cwd=ROOT,env=env)
    raise SystemExit(result.returncode)
if __name__=='__main__':main()
