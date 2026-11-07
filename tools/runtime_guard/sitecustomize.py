"""Accidental-write guard, activated only by the Route-A launcher."""
import os,sys
from pathlib import Path
if os.environ.get('ROUTE_A_GUARD')=='1':
    _root=Path(os.environ['ROUTE_A_PROJECT_ROOT']).resolve()
    def _check(path):
        if isinstance(path,int) or path is None:return
        p=Path(os.fsdecode(path)).resolve()
        if not p.is_relative_to(_root):raise PermissionError('ROUTE_A_WRITE_BLOCKED: '+str(p))
    def _guard(event,args):
        if event=='open':
            path,mode,flags=args
            if (isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)):_check(path)
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','os.truncate'):_check(args[0])
        elif event in ('os.rename','os.replace'):_check(args[0]);_check(args[1])
        elif event in ('os.symlink','os.link'):raise PermissionError('ROUTE_A_LINK_CREATION_BLOCKED')
    sys.addaudithook(_guard)
